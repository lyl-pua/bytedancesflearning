# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路里的位置：
#   Agent 建好模型后，还要告诉模型「你有哪些工具可以调用」
#   → 工具的清单和开关来自 config.yaml 的 tools: 段 + MCP 服务器 + 内置工具
#   → 这里把它们按条件筛出来、去重、补齐，返回一个 BaseTool 列表绑定给模型。
#   所以本文件是「工具装配车间」：它不实现工具本身，只决定「这次给模型哪些」。
#
# ★ 它解决什么问题：
#   1. 按配置分组(group)筛选工具，并可关闭某些危险能力（比如宿主机 bash）；
#   2. 根据运行环境动态追加工具（是否支持调度、子 agent、上传、图片理解…）；
#   3. 合并 MCP（Model Context Protocol）远程工具，并处理重名冲突；
#   4. 按名字去重——重名会让模型收到歧义的函数定义，从而调用失败。
#
# ● 涉及的知识点：
#   1. Function Calling / tool_calls：模型不是直接执行代码，而是「请求调用某个工具」，
#      所以工具的 name + description（schema）必须唯一且清晰。
#   2. 动态 import（importlib）：resolve_variable 按配置字符串加载工具对象。
#   3. 同步 vs 异步工具：LangChain 工具有 coroutine 时是异步的，
#      但同步调用链需要 func，所以要补一个同步包装（见 _ensure_sync_invocable_tool）。
#   4. 双检锁(double-checked locking)：同一工具对象被多线程装配时，只包一次。
#   5. 不可变共享状态的坑：模块级单例工具如果被就地修改，会污染所有调用方，
#      所以需要「克隆后改描述」。
#   6. MCP：一种把外部工具服务器统一暴露给模型的标准协议。

import copy
import logging
import threading

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.constants import CONVERSATION_TOOL_USE
from deerflow.mcp.tasks.runtime import is_mcp_task_runtime_available
from deerflow.reflection import resolve_variable
from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.scheduler.runtime import SchedulerRunCapability, is_scheduler_capability, scheduler_tools_enabled
from deerflow.subagents.batch_runtime import is_subagent_batch_runtime_available
from deerflow.tools.builtins import (
    ask_clarification_tool,
    build_inspect_run_status_tool,
    cancel_background_task,
    list_background_tasks,
    list_uploaded_files,
    present_file_tool,
    review_skill_package,
    task_tool,
    view_image_tool,
)
from deerflow.tools.mcp_metadata import get_mcp_source, tag_mcp_tool
from deerflow.tools.sync import make_sync_tool_wrapper

logger = logging.getLogger(__name__)

# 三个「无条件内置」的小工具：给用户展示文件、向用户追问、审核技能包。
# 它们不依赖任何配置，所以是常量列表，直接复用同一批对象（因此后面要小心不要就地修改）。
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    review_skill_package,
]

# 子 agent 只能拿到「派活」这一个工具：如果给子 agent 也开放全部工具，
# 会导致递归失控与权限扩散，所以这里刻意收窄。
SUBAGENT_TOOLS = [
    task_tool,
    # task_status_tool is no longer exposed to LLM (backend handles polling internally)
]


# ★ 典型的「危险能力识别」：判断一个工具配置指向的是不是「在宿主机执行 bash」。
# 同时比对 group 与 use 两个字段，是因为配置作者可能只写了一处，
# 只按 group 判断会漏掉直接写 use 的写法（那就等于给了一次绕过安全检查的机会）。
def _is_host_bash_tool(tool: object) -> bool:
    """Return True if the tool config represents a host-bash execution surface."""
    group = getattr(tool, "group", None)
    use = getattr(tool, "use", None)
    if group == "bash":
        return True
    if use == "deerflow.sandbox.tools:bash_tool":
        return True
    return False


# 保护「给异步工具补同步包装」这个动作的锁。工具对象是进程级单例，
# 装配又可能并行发生在多个线程上，所以包裹动作必须是幂等的(single-shot)。
_sync_invocable_tool_lock = threading.Lock()


# ★ 双检锁(double-checked locking)：先不加锁快速判断（绝大多数工具已经有 func，直接返回），
# 只有需要补包装时才进锁；进锁后再检查一次，防止别的线程已经包好了，这里重复包。
# 为什么值得这么做：工具装配在请求路径上，能省一把锁就省一次串行化。
def _ensure_sync_invocable_tool(tool: BaseTool) -> BaseTool:
    """Attach a sync wrapper to async-only tools used by sync agent callers.

    The wrapped objects are process-wide singletons (BUILTIN_TOOLS /
    SUBAGENT_TOOLS / MCP cache entries) and tool assembly may now run on
    worker threads concurrently; double-checked locking makes the in-place
    ``tool.func`` wrap explicitly single-shot instead of incidental.
    """
    if getattr(tool, "func", None) is not None or getattr(tool, "coroutine", None) is None:
        return tool
    with _sync_invocable_tool_lock:
        if getattr(tool, "func", None) is None:
            tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)
    return tool


# 从模型配置里安全地取 max_tokens。写这么多判断是因为配置对象来源五花八门
# （Pydantic 模型 / dict / 测试桩），而且 bool 在 Python 里是 int 的子类，
# 不显式排除 bool 的话 True 会被当作 1 传下去。
def _extract_max_tokens(model_config: object | None) -> int | None:
    """Safely extract a positive integer max_tokens from a model config object.

    Handles ModelConfig (where max_tokens may be stored as an extra dynamic field),
    dicts, SimpleNamespace, or test stubs. Rejects booleans, mocks, non-numeric
    values, non-finite numbers, negative numbers, zero, and None.
    """
    if model_config is None:
        return None
    raw = model_config.get("max_tokens") if isinstance(model_config, dict) else getattr(model_config, "max_tokens", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


# 「克隆后再改」而不是直接改原对象：tool 是模块级单例，
# 就地改 description 会把预算提示永久写进这个工具，影响所有模型的所有会话。
def _clone_tool_with_description(tool: BaseTool, description: str) -> BaseTool:
    """Return a copy of tool with an updated description, leaving the original intact."""
    if isinstance(tool, BaseModel):
        return tool.model_copy(update={"description": description})
    cloned = copy.copy(tool)
    cloned.description = description
    return cloned


# ★★★ 主入口：装配「本次请求该给模型哪些工具」。
# 阅读顺序建议：
#   1) 按 group 过滤配置 → 2) 按运行能力剔除危险工具 → 3) 动态加载配置里的工具
#   → 4) 追加内置工具（订阅条件） → 5) 合并 MCP 工具 → 6) 加 ACP/插件工具 → 7) 按名去重
def get_available_tools(
    groups: list[str] | None = None,
    include_mcp: bool = True,
    model_name: str | None = None,
    subagent_enabled: bool = False,
    *,
    mcp_plugins: list[str] | None = None,
    include_upload_tool: bool = True,
    include_conversation_reader: bool = False,
    scheduler_capability: SchedulerRunCapability | None = None,
    app_config: AppConfig | None = None,
    extensions=None,
    chat_model: BaseChatModel | None = None,
) -> list[BaseTool]:
    """Get all available tools from config.

    Note: MCP tools should be initialized at application startup using
    `initialize_mcp_tools()` from deerflow.mcp module.

    Args:
        groups: Optional list of tool groups to filter by.
        include_mcp: Whether to include tools from MCP servers (default: True).
        model_name: Optional model name to determine if vision tools should be included.
        chat_model: Constructed model whose effective output cap supplies write_file
            guidance. When supplied, an absent cap omits the hint; only callers
            without a model fall back to the configured profile.
        subagent_enabled: Whether to include subagent tools (task, task_status).
        include_upload_tool: Whether to include ``list_uploaded_files`` (default: True).
            Ordinary task subagents enable it only after snapshotting the
            parent's current-run upload state. Durable batch and non-standard
            subagent callers without that state keep it disabled.
        include_conversation_reader: Allow the configured conversation reader
            only when the host provides its authorized runtime capability.
            Defaults to false for embedded callers and subagents.
        scheduler_capability: Current-run host grant for conversation schedule
            management or own-schedule stopping. Omitted for embedded,
            bootstrap and subagent assembly.

    Returns:
        List of available tools.
    """
    config = app_config or get_app_config()
    # 第一道筛：按 group 过滤。groups 为 None 表示「全都要」。
    tool_configs = [tool for tool in config.tools if groups is None or tool.group in groups]
    # These operations are assembled from the host grant below. Registering a
    # tool path in YAML cannot widen that grant or its interaction mode.
    tool_configs = [tool for tool in tool_configs if tool.use not in {"deerflow.tools.scheduled_tasks:schedule_task", "deerflow.tools.scheduled_tasks:stop_scheduled_task"}]
    if not include_conversation_reader:
        tool_configs = [tool for tool in tool_configs if tool.use != CONVERSATION_TOOL_USE]

    # Knowledge tools are opt-in as a group. Provider connection and retrieval
    # settings live on each tool entry; the generic capability flag controls
    # whether the group is exposed at all.
    knowledge_base_config = getattr(config, "knowledge_base", None)
    if not getattr(knowledge_base_config, "enabled", False):
        tool_configs = [tool for tool in tool_configs if tool.group != "knowledge"]

    # Do not expose host bash by default when LocalSandboxProvider is active.
    # ★ 安全兜底的思路（路径穿越/越权防护的同类思想）：YAML 里写什么都不该能扩大运行时的授权。
    # 每次装配都先剔除宿主机 bash —— 除非安全策略明确允许（is_host_bash_allowed）。
    if not is_host_bash_allowed(config):
        tool_configs = [tool for tool in tool_configs if not _is_host_bash_tool(tool)]

    # ★ 动态加载：config 里写的是字符串路径，resolve_variable 在运行期 import 出真正的工具对象，
    # 并校验它是 BaseTool —— 这样新增工具只需要改配置，不用改这段代码。
    loaded_tools_raw = [(cfg, resolve_variable(cfg.use, BaseTool)) for cfg in tool_configs]

    # Warn when the config ``name`` field and the tool object's ``.name``
    # attribute diverge — this mismatch is the root cause of issue #1803 where
    # the LLM receives one name in its tool schema but the runtime router
    # recognises a different name, producing "not a valid tool" errors.
    for cfg, loaded in loaded_tools_raw:
        if cfg.name != loaded.name:
            logger.warning(
                "Tool name mismatch: config name %r does not match tool .name %r (use: %s). The tool's own .name will be used for binding.",
                cfg.name,
                loaded.name,
                cfg.use,
            )

    # 统一补齐「同步可调用」能力，因为上层既可能是 async 也可能是 sync 调用链。
    loaded_tools = [_ensure_sync_invocable_tool(t) for _, t in loaded_tools_raw]

    # Conditionally add tools based on config
    builtin_tools = BUILTIN_TOOLS.copy()
    if scheduler_tools_enabled(config) and is_scheduler_capability(scheduler_capability):
        from deerflow.tools.scheduled_tasks import schedule_task, stop_scheduled_task

        builtin_tools.append(schedule_task if scheduler_capability.mode == "interactive" else stop_scheduled_task)
    if is_mcp_task_runtime_available():
        builtin_tools.extend((list_background_tasks, cancel_background_task))
    if include_upload_tool:
        builtin_tools.append(list_uploaded_files)
    skill_evolution_config = getattr(config, "skill_evolution", None)
    if getattr(skill_evolution_config, "enabled", False):
        from deerflow.tools.skill_manage_tool import skill_manage_tool

        builtin_tools.append(skill_manage_tool)

    # Add subagent tools only if enabled via runtime parameter
    if subagent_enabled:
        builtin_tools.extend(SUBAGENT_TOOLS)
        if is_subagent_batch_runtime_available():
            from deerflow.subagents.batch_runtime import get_subagent_batch_submitter
            from deerflow.tools.builtins.batch_task_tool import bind_batch_tools

            builtin_tools.extend(bind_batch_tools(submitter_provider=get_subagent_batch_submitter, app_config=config))
        logger.info("Including native subagent tools")

    # If no model_name specified, use the first model (default)
    if model_name is None and config.models:
        model_name = config.models[0].name

    # Add view_image_tool only if the model supports vision
    model_config = config.get_model_config(model_name) if model_name else None
    builtin_tools.append(build_inspect_run_status_tool(context_window=model_config.context_window if model_config else None))
    # ★ 这里临时 import 而不是放到文件顶部，是刻意的「懒加载」：
    # 只有真的开了这个能力才 import，既能避免循环 import，也能减少启动开销。
    if model_config is not None and model_config.supports_vision:
        builtin_tools.append(view_image_tool)
        logger.info(f"Including view_image_tool for model '{model_name}' (supports_vision=True)")

    # Annotate write_file with the constructed model's effective output budget so the
    # model does not assume the 80 KB streaming ceiling is the practical limit
    # for a single completion. The tool is cloned to avoid mutating the
    # module-level singleton in-place across assemblies or leaking guidance to
    # models configured without max_tokens.
    # ★ 「给模型的工具说明书加一句预算提示」：
    # 模型并不知道自己一次最多能输出多少 token，于是经常一次性写超大文件被截断。
    # 这里把上限换算成大致字符数写进 write_file 的描述里，引导它分段写 + append。
    max_tokens = _extract_max_tokens(chat_model if chat_model is not None else model_config)
    if max_tokens is not None:
        try:
            safe_chars = int(max_tokens * 3 * 0.7)
        except OverflowError:
            # Even finite caps can be too large for this optional estimate.
            pass
        else:
            budget_note = (
                f"\n\nPER-RESPONSE BUDGET: your output limit is {max_tokens} tokens "
                f"(≈{safe_chars} chars). Single non-append writes above this will be truncated. "
                "For larger documents, write the first section now, "
                "then use append=True for subsequent sections."
            )
            loaded_tools = [
                _clone_tool_with_description(
                    tool,
                    f"{getattr(tool, 'description', '') or ''}{budget_note}",
                )
                if tool.name == "write_file" and hasattr(tool, "description") and "PER-RESPONSE BUDGET:" not in (getattr(tool, "description", "") or "")
                else tool
                for tool in loaded_tools
            ]

    # Get cached MCP tools if enabled
    # NOTE: We use ExtensionsConfig.from_file() instead of config.extensions
    # to always read the latest configuration from disk. This ensures that changes
    # made through the Gateway API (which runs in a separate process) are immediately
    # reflected when loading MCP tools.
    # 每个 MCP 工具都要打上「来源标签」(server_name)，后续做插件过滤/权限判断时靠它区分归属。
    # 不标的话，所有工具看起来都像是本地内置的，无法按来源做管控。
    mcp_tools = []
    if include_mcp:
        try:
            from deerflow.config.extensions_config import ExtensionsConfig
            from deerflow.mcp.cache import get_cached_mcp_tools, refresh_mcp_cache_if_active

            try:
                extensions_config = ExtensionsConfig.from_file()
            except Exception as exc:
                # Only this call carries the resolved-credential risk:
                # from_file() resolves $VAR values before validation, so a
                # ValidationError message can embed secrets. Log the type only.
                logger.error("Failed to load MCP extensions config (%s)", type(exc).__name__)
            else:
                if extensions_config.get_enabled_mcp_servers():
                    mcp_tools = get_cached_mcp_tools()
                    if mcp_tools:
                        logger.info(f"Using {len(mcp_tools)} cached MCP tool(s)")

                        # Tag MCP-sourced tools so deferred-tool assembly at each
                        # agent construction site can identify them. Lead agents
                        # assemble their full configured MCP catalog; subagents
                        # may pass an already filtered list reflecting their own
                        # tool configuration and authorization.
                        for t in mcp_tools:
                            tag_mcp_tool(t)
                else:
                    # A change that disables the last MCP server must still retire
                    # the previously initialized cache and its pooled sessions.
                    # This never initializes tools: a process that never initialized
                    # MCP pays no config-hashing or discovery cost, while one that
                    # did still checks the existing cache for staleness.
                    refresh_mcp_cache_if_active()
                from deerflow.mcp.user_tools import get_user_mcp_tools

                try:
                    personal_tools, personal_config = get_user_mcp_tools()
                except Exception as exc:
                    # Invalid personal credentials must neither be logged nor
                    # bypass explicit selection of the platform tools.
                    logger.warning("Could not load personal MCP tools (%s)", type(exc).__name__)
                    personal_tools, personal_config = [], ExtensionsConfig()
                colliding_names = extensions_config.mcp_servers.keys() & personal_config.mcp_servers.keys()
                # 个人 MCP 与部署级 MCP 可能出现同名 server；
                # 此时保留部署级的（可信），丢掉同名的个人工具，避免路由串到错误的服务器。
                if colliding_names:
                    # A deployment name can match a generated personal runtime
                    # name. Keep only the deployment tool and its routing metadata.
                    logger.warning("Skipping personal MCP tools with deployment name collisions: %s", sorted(colliding_names))
                    personal_tools = [tool for tool in personal_tools if (source := get_mcp_source(tool)) and source["server_name"] not in colliding_names]
                    available_servers = {name: server for name, server in personal_config.mcp_servers.items() if name not in colliding_names}
                    personal_config = personal_config.model_copy(update={"mcp_servers": available_servers})
                mcp_tools = [*mcp_tools, *personal_tools]
                extensions_config = extensions_config.model_copy(update={"mcp_servers": {**extensions_config.mcp_servers, **personal_config.mcp_servers}})
                if mcp_plugins is not None:
                    from deerflow.capabilities.runtime import filter_mcp_plugins

                    mcp_tools = filter_mcp_plugins(mcp_tools, mcp_plugins, extensions_config)
        except ImportError:
            logger.warning("MCP module not available. Install 'langchain-mcp-adapters' package to enable MCP tools.")
        except Exception:
            # Tool caching, pool cleanup and plugin filtering raise ordinary
            # exceptions whose messages are safe and needed for debugging.
            logger.exception("Failed to get cached MCP tools")

    # Add invoke_acp_agent tool if any ACP agents are configured
    acp_tools: list[BaseTool] = []
    try:
        from deerflow.tools.builtins.invoke_acp_agent_tool import build_invoke_acp_agent_tool

        if app_config is None:
            from deerflow.config.acp_config import get_acp_agents

            acp_agents = get_acp_agents()
        else:
            acp_agents = getattr(config, "acp_agents", {}) or {}
        if acp_agents:
            acp_tools.append(build_invoke_acp_agent_tool(acp_agents))
            logger.info(f"Including invoke_acp_agent tool ({len(acp_agents)} agent(s): {list(acp_agents.keys())})")
    except Exception as e:
        logger.warning(f"Failed to load ACP tool: {e}")

    logger.info(f"Total tools loaded: {len(loaded_tools)}, built-in tools: {len(builtin_tools)}, MCP tools: {len(mcp_tools)}, ACP tools: {len(acp_tools)}")

    # Deduplicate by tool name — config-loaded tools take priority, followed by
    # built-ins, MCP tools, and ACP tools.  Duplicate names cause the LLM to
    # receive ambiguous or concatenated function schemas (issue #1803).
    from deerflow.extensions import get_agent_build_extensions
    from deerflow.extensions.plugin_tools import build_plugin_tools

    # ★ 最终去重：按工具名保留第一个出现的（优先级：配置 > 内置 > MCP > ACP > 插件）。
    # 为什么必须做：同名工具会让模型收到两份 schema，它可能挑错那个，
    # 运行时路由又只认其中一个，于是报「not a valid tool」这种很难查的错误。
    ordinary_tools = loaded_tools + builtin_tools + mcp_tools + acp_tools
    # Keep plugin-vs-plugin validation strict. Host/plugin collisions use the
    # ordinary-first deduplication below, without dropping unrelated tools.
    plugin_tools = build_plugin_tools(extensions if extensions is not None else get_agent_build_extensions(), groups=groups)
    all_tools = [_ensure_sync_invocable_tool(t) for t in ordinary_tools + plugin_tools]
    # 用 set 记录已出现的名字，达到 O(n) 去重；这里顺便把插件工具也补齐同步能力。
    seen_names: set[str] = set()
    unique_tools: list[BaseTool] = []
    for t in all_tools:
        if t.name not in seen_names:
            unique_tools.append(t)
            seen_names.add(t.name)
        else:
            logger.warning(
                "Duplicate tool name %r detected and skipped — check your config.yaml and MCP server registrations (issue #1803).",
                t.name,
            )
    return unique_tools
