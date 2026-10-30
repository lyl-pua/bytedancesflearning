# =============================================================================
# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路的位置
#   最底层：langchain.agents.create_agent（LangChain 官方「图构造器」原语）
#   本文件：create_deerflow_agent（SDK 层装配函数，只吃「纯 Python 参数」）
#   上层：lead_agent/agent.py 的 make_lead_agent（读 config.yaml 的应用级装配）
#
#   一句话：本文件负责把「模型 + 工具 + 一排中间件 + 状态结构」交给
#   LangChain，换回一张可以 invoke 的 Agent 图（CompiledStateGraph）。
#   它刻意不读任何配置文件，方便被 SDK 使用者、测试、批处理直接调用。
#
# ★ 它解决什么问题
#   手工拼装 Agent 时顺序极易搞错（例如：摘要中间件必须在其它处理之前、
#   澄清中间件必须永远在最后、沙盒中间件必须在上传中间件之前……）。
#   本文件把这些「顺序约束」固化成一份声明式清单
#   （RuntimeFeatures → 中间件链），调用方只需说「我要 memory、要
#   subagent、要 vision」，不用关心谁先谁后。
#
# ● 涉及的知识点（编号说明）
#   1. StateGraph / Agent 图：LangGraph 用「图」表达 Agent 循环：
#      模型节点 → 工具节点 → 再回模型，直到模型不再要求调工具。
#      create_agent 就是帮你生成这张固定骨架图的工厂。
#   2. 节点（node）= 图里的一个处理步骤；边（edge）= 步骤之间的跳转；
#      条件边（conditional edge）= 依据模型输出（有没有 tool_calls）决定下一步。
#      本文件不直接搭图，只把「每步前后要跑的钩子」交给中间件机制。
#   3. 中间件链（middleware chain）：可理解为「环绕模型调用的洋葱」。
#      每个中间件可在模型调用前（before_model）改请求、在模型调用后
#      （after_model）改结果。顺序 = 洋葱的层序，越靠前越靠外。
#   4. 上下文工程（context engineering）：不靠改模型权重，而是靠
#      「喂给模型的上下文里放什么、放多少、怎么压缩」来提升表现。
#      中间件链就是上下文工程的主要落点（摘要、记忆注入、token 预算……）。
#   5. token 预算与上下文压缩：模型的上下文窗口有上限（比如 128k token），
#      历史越长越贵、越容易超限，所以需要「摘要压缩」把老消息折叠成一段
#      summary_text（见 summarization_middleware.py）。
#   6. checkpointer 与 thread_id：checkpointer 负责把每步状态存下来（类似
#      存档点），thread_id 是「存档槽位编号」；同一 thread_id 重放即可续聊。
#      checkpoint_channel_mode 决定「存全量快照」还是「存增量（delta）」。
#   7. 依赖注入（Dependency Injection）：本文件的参数全是显式传入，
#      不读全局单例，因此可以在一个进程里并存多个互不干扰的图。
# =============================================================================

"""Pure-argument factory for DeerFlow agents.

``create_deerflow_agent`` accepts plain Python arguments — it does not load
YAML or install process-global runtime dependencies. It is the SDK-level entry
point sitting between the raw
``langchain.agents.create_agent`` primitive and the config-driven
``make_lead_agent`` application factory.

Direct callers that need an isolated native-subagent capacity or a durable
batch worker pass a caller-owned ``SubagentRuntime`` explicitly. When omitted,
subagent tools retain their application-compatible process-global fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from deerflow.agents.features import RuntimeFeatures
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
from deerflow.agents.thread_state import adapt_state_schema_for_mode, get_thread_state_schema, normalize_middleware_state_schemas
from deerflow.config.database_config import CheckpointChannelMode
from deerflow.tools.builtins import ask_clarification_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from deerflow.config.memory_config import MemoryConfig
    from deerflow.config.pii_redaction_config import PiiRedactionConfig
    from deerflow.subagents.runtime import SubagentRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TodoMiddleware prompts (minimal SDK version)
# ---------------------------------------------------------------------------

# ★ 为什么这份 todo 提示词要在 SDK 层「内置一份最小版」？
#   上面那些配置驱动的装配可以自由传自定义提示词，但 SDK 调用方不该为了
#   用 plan_mode 就去复制一份长提示词。这里放一份最小可用版本，保证
#   create_deerflow_agent(plan_mode=True) 开箱即用；上层应用可以覆盖它。
_TODO_SYSTEM_PROMPT = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly
</todo_list_system>
"""

_TODO_TOOL_DESCRIPTION = "Use this tool to create and manage a structured task list for complex work sessions.  Only use for complex tasks (3+ steps)."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ★ 装配总流程（初学者按这 6 步读代码即可）
#   第1步 参数体检：middleware 全接管 与 features/extra_middleware 不能混用；
#          delta 检查点模式与 checkpointer 不能混用（否则会静默写坏 thread 状态）。
#   第2步 决定「状态结构」：ThreadState（或调用方自定义的 state_schema），
#          delta 模式下会换成 DeltaThreadState（消息通道改为增量存储）。
#   第3步 决定「中间件链」：要么完全用调用方给的 middleware（全接管），
#          要么按 RuntimeFeatures 声明式装配（_assemble_from_features）。
#   第4步 合并工具：用户工具优先，feature 额外注入的工具按名字去重追加。
#   第5步 让中间件里声明的 state_schema 也适配 delta 模式。
#   第6步 交给 create_agent，得到一张可运行的图。
#
# ● 关键概念：为什么有「全接管」和「声明式」两种模式？
#   全接管 = 调用方自己负责顺序，框架不做任何假设（灵活但易错）；
#   声明式 = 框架保证顺序约束（安全，但只能按既有清单来）。
#   两者互斥，是为了避免「你以为自己控制了顺序，其实被框架又插了一层」。
def create_deerflow_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    features: RuntimeFeatures | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
    plan_mode: bool = False,
    state_schema: type | None = None,
    checkpoint_channel_mode: CheckpointChannelMode = "full",
    checkpoint_snapshot_frequency: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
    subagent_runtime: SubagentRuntime | None = None,
    pii_redaction_config: PiiRedactionConfig | None = None,
) -> CompiledStateGraph:
    """Create a DeerFlow agent from plain Python arguments.

    The factory assembly itself reads no config files. Pass ``subagent_runtime``
    when direct SDK-created graphs must share an explicit native-subagent
    capacity or caller-managed durable batch worker.

    Parameters
    ----------
    model:
        Chat model instance.
    tools:
        User-provided tools.  Feature-injected tools are appended automatically.
    system_prompt:
        System message.  ``None`` uses a minimal default.
    middleware:
        **Full takeover** — if provided, this exact list is used.
        Cannot be combined with *features* or *extra_middleware*.
    features:
        Declarative feature flags.  Cannot be combined with *middleware*.
    extra_middleware:
        Additional middlewares inserted into the auto-assembled chain via
        ``@Next``/``@Prev`` positioning.  Cannot be used with *middleware*.
    plan_mode:
        Enable TodoMiddleware for task tracking.
    state_schema:
        LangGraph state type.  Defaults to ``ThreadState``.
    checkpoint_channel_mode:
        Checkpoint representation for accumulating channels.  Defaults to the
        full-state compatibility schema.  ``"delta"`` requires the guarded
        persistence paths (mode markers + compatibility gate) and is therefore
        rejected when combined with *checkpointer* in this factory; without a
        checkpointer the graph is ephemeral and delta is allowed.
    checkpoint_snapshot_frequency:
        DeltaChannel snapshot cadence for ``"delta"`` mode.  ``None`` uses the
        process-frozen value, falling back to the config default.  Ignored in
        ``"full"`` mode.
    checkpointer:
        Optional persistence backend.
    name:
        Agent name (passed to middleware that cares, e.g. ``MemoryMiddleware``).
    subagent_runtime:
        Explicit process runtime shared by direct SDK-created graphs. Required
        only when the caller needs non-default native-subagent capacity or a
        caller-managed durable batch worker without Gateway/DeerFlowClient
        startup. Requires ``features.subagent`` to be enabled.
    pii_redaction_config:
        Optional PII redaction policy (#3190). ``None`` leaves redaction off;
        when enabled, memory-queue and durable-context redaction apply to the
        assembled middlewares. A ``middleware=`` takeover bypasses the
        assembly that would apply it.

    Raises
    ------
    ValueError
        If both *middleware* and *features*/*extra_middleware* are provided.
    """
    # -------------------------------------------------------------------
    # 第1步：参数体检（fail fast）
    # 这些都是「配置错误」，与其让它们在运行几轮后才以诡异形式爆炸，
    # 不如在装配时立刻抛 ValueError，让人一眼看出是调用姿势错了。
    # -------------------------------------------------------------------
    if middleware is not None and features is not None:
        raise ValueError("Cannot specify both 'middleware' and 'features'.  Use one or the other.")
    if checkpoint_channel_mode == "delta" and checkpointer is not None:
        raise ValueError(
            "create_deerflow_agent does not support checkpoint_channel_mode='delta' with a checkpointer: "
            "persisted graphs built here bypass checkpoint mode marker injection and the fail-closed "
            "compatibility gate (see deerflow.runtime.checkpoint_mode), so a mixed-mode store would "
            "silently corrupt thread state.  Use the guarded application paths (make_lead_agent or "
            "DeerFlowClient) for delta persistence; delta without a checkpointer is ephemeral and allowed."
        )
    if middleware is not None and extra_middleware:
        raise ValueError("Cannot use 'extra_middleware' with 'middleware' (full takeover).")
    if subagent_runtime is not None and (middleware is not None or features is None or features.subagent is False):
        raise ValueError("subagent_runtime requires features.subagent to be enabled; it cannot be used with middleware full takeover")
    if subagent_runtime is not None and subagent_runtime.batch_config is not None and subagent_runtime.batch_submitter is None:
        raise RuntimeError("The explicit durable batch worker is not running; await subagent_runtime.start() or enter it with 'async with' before calling create_deerflow_agent")
    if extra_middleware:
        for mw in extra_middleware:
            if not isinstance(mw, AgentMiddleware):
                raise TypeError(f"extra_middleware items must be AgentMiddleware instances, got {type(mw).__name__}")

    # -------------------------------------------------------------------
    # 第2步：确定状态结构（StateGraph 的「状态」= 全图共享的一块数据黑板）
    # 每个节点返回的东西都会按 reducer 合并进这块黑板；state_schema 决定
    # 黑板上有哪些字段、每个字段怎么合并。delta 模式下 messages 字段会被
    # 换成 DeltaChannel（只存增量，节省检查点体积）。
    # -------------------------------------------------------------------
    effective_tools: list[BaseTool] = list(tools or [])
    effective_state = get_thread_state_schema(checkpoint_channel_mode, checkpoint_snapshot_frequency) if state_schema is None else adapt_state_schema_for_mode(state_schema, checkpoint_channel_mode, checkpoint_snapshot_frequency)

    # -------------------------------------------------------------------
    # 第3步 + 第4步：装配中间件链、合并工具
    # 注意：用户自己传的工具优先（同名时跳过 feature 注入的那个），
    # 这样调用方的显式选择永远能覆盖框架的默认，避免「悄悄被换掉」。
    # -------------------------------------------------------------------
    if middleware is not None:
        effective_middleware = list(middleware)
    else:
        feat = features or RuntimeFeatures()
        effective_middleware, extra_tools = _assemble_from_features(
            feat,
            name=name,
            plan_mode=plan_mode,
            extra_middleware=extra_middleware or [],
            subagent_runtime=subagent_runtime,
            pii_redaction_config=pii_redaction_config,
        )
        # Deduplicate by tool name — user-provided tools take priority.
        existing_names = {t.name for t in effective_tools}
        for t in extra_tools:
            if t.name not in existing_names:
                effective_tools.append(t)
                existing_names.add(t.name)

    # 第5步：有些中间件自己带 state_schema，也要跟着 delta 模式换一次，
    # 否则「中间件按全量 schema 写状态、图按增量 schema 读状态」就会对不上。
    effective_middleware = normalize_middleware_state_schemas(
        effective_middleware,
        checkpoint_channel_mode,
        checkpoint_snapshot_frequency,
    )

    # 第6步：真正交给 LangChain 生成图。
    # ★ 这里不传 checkpointer 以外的任何「持久化」参数：图的持久化语义
    #   完全由 checkpointer + state_schema 决定，本函数不掺和。
    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )


# ---------------------------------------------------------------------------
# Internal: feature-driven middleware assembly
# ---------------------------------------------------------------------------


# ★ 声明式装配的核心：把「功能开关」翻译成「有序中间件链 + 额外工具」
#   下面 docstring 里的编号清单就是本项目的「顺序宪法」，读代码时可当作目录。
# ● 为什么顺序如此重要（举三个例子，都是真实踩过的坑）：
#   - 沙盒/上传中间件必须在前：后面的中间件都要用到 thread_id、
#     工作目录路径，这些由它们先准备好。
#   - 摘要（summarization）必须靠前：先把上下文压小，后面的中间件
#     才不会在超长历史上做昂贵的处理。
#   - 澄清（clarification）必须永远最后：它要在模型已经吐出「我需要
#     问用户」的信号之后才拦截，放中间会截断流程。
# ● 还体现了「渐进式披露（progressive disclosure）」的工程实现方式：
#   工具和技能的完整说明不一次性塞给模型，而是先给索引、按需展开
#   （见 token 预算 / tool_search / skill 索引相关中间件）。
def _assemble_from_features(
    feat: RuntimeFeatures,
    *,
    name: str = "default",
    plan_mode: bool = False,
    extra_middleware: list[AgentMiddleware] | None = None,
    subagent_runtime: SubagentRuntime | None = None,
    pii_redaction_config: PiiRedactionConfig | None = None,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """Build an ordered middleware chain + extra tools from *feat*.

    Middleware order matches ``make_lead_agent`` (18 middlewares):

      0-2. Sandbox infrastructure (ThreadData → Uploads → Sandbox)
      3.   DanglingToolCallMiddleware (always)
      4.   GuardrailMiddleware (guardrail feature)
      5.   ToolErrorHandlingMiddleware (always)
      5a.  DurableContextMiddleware (always)
      5b.  SystemMessageCoalescingMiddleware (always)
      6.   SummarizationMiddleware (summarization feature)
      7.   TodoMiddleware (plan_mode parameter)
      8.   TitleMiddleware (auto_title feature)
      9.   MemoryMiddleware (memory feature)
      10.  ViewImageMiddleware (vision feature)
      11.  SubagentLimitMiddleware (subagent feature)
      12.  LoopDetectionMiddleware (loop_detection feature)
      13.  TokenBudgetMiddleware (token_budget feature)
      13a. ModelContentCompatibilityMiddleware (always)
      14.  ClarificationMiddleware (always last)

    Two-phase ordering:
      1. Built-in chain — fixed sequential append.
      2. Extra middleware — inserted via @Next/@Prev.

    Each feature value is handled as:
      - ``False``: skip
      - ``True``: create the built-in default middleware (not available for
        ``summarization`` and ``guardrail`` — these require a custom instance)
      - ``AgentMiddleware`` instance: use directly (custom replacement)
    """
    # 每种 feature 的取值有三种语义（docstring 也写了）：
    #   False → 不要；True → 用框架内置默认；传实例 → 用调用方的定制版。
    # 这让「关闭 / 默认 / 自定义」三态无需额外的 None 哨兵值。
    chain: list[AgentMiddleware] = []
    extra_tools: list[BaseTool] = []

    # --- [0-2] Sandbox infrastructure ---
    if feat.sandbox is not False:
        if isinstance(feat.sandbox, AgentMiddleware):
            chain.append(feat.sandbox)
        else:
            from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
            from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware
            from deerflow.sandbox.middleware import SandboxMiddleware

            chain.append(ThreadDataMiddleware(lazy_init=True))
            chain.append(UploadsMiddleware())
            chain.append(SandboxMiddleware(lazy_init=True))

    # --- [3] DanglingToolCall (always) ---
    chain.append(DanglingToolCallMiddleware())

    # --- [4] Guardrail ---
    if feat.guardrail is not False:
        if isinstance(feat.guardrail, AgentMiddleware):
            chain.append(feat.guardrail)
        else:
            raise ValueError("guardrail=True requires a custom AgentMiddleware instance (no built-in GuardrailMiddleware yet)")

    # --- [5] ToolErrorHandling (always) ---
    chain.append(ToolErrorHandlingMiddleware())

    # --- [5a] DurableContext (always) ---
    # Summarization moves compacted history into ``summary_text``, and
    # SubagentLimitMiddleware counts the run's delegations from the
    # ``delegations`` ledger. This middleware writes that ledger and projects
    # both into model requests. It sits ahead of summarization, as in
    # make_lead_agent, so delegations are captured before they are compacted.
    from deerflow.agents.middlewares.durable_context_middleware import DurableContextMiddleware
    from deerflow.agents.middlewares.system_message_coalescing_middleware import SystemMessageCoalescingMiddleware

    if pii_redaction_config is not None and pii_redaction_config.enabled:
        from deerflow.agents.middlewares.pii_redaction_middleware import PiiRedactionMiddleware

        chain.append(PiiRedactionMiddleware(pii_redaction_config))

    chain.append(DurableContextMiddleware(pii_redaction_config=pii_redaction_config))
    # DurableContext adds its authority contract as a second SystemMessage; strict backends
    # (vLLM, SGLang, Qwen, Anthropic) reject that, so merge them into one leading message.
    chain.append(SystemMessageCoalescingMiddleware())

    # --- [6] Summarization ---
    if feat.summarization is not False:
        if isinstance(feat.summarization, AgentMiddleware):
            chain.append(feat.summarization)
        else:
            raise ValueError("summarization=True requires a custom AgentMiddleware instance (SummarizationMiddleware needs a model argument)")

    # --- [7] TodoMiddleware (plan_mode) ---
    if plan_mode:
        from deerflow.agents.middlewares.todo_middleware import TodoMiddleware

        chain.append(TodoMiddleware(system_prompt=_TODO_SYSTEM_PROMPT, tool_description=_TODO_TOOL_DESCRIPTION))

    # --- [8] Auto Title ---
    if feat.auto_title is not False:
        if isinstance(feat.auto_title, AgentMiddleware):
            chain.append(feat.auto_title)
        else:
            from deerflow.agents.middlewares.title_middleware import TitleMiddleware

            chain.append(TitleMiddleware())

    # --- [9] Memory ---
    if feat.memory is not False:
        if isinstance(feat.memory, AgentMiddleware):
            chain.append(feat.memory)
        else:
            from deerflow.config.memory_config import get_memory_config, should_use_memory_tools

            memory_cfg: MemoryConfig = feat.memory_config or get_memory_config()
            if should_use_memory_tools(memory_cfg):
                from deerflow.agents.memory.manager import backend_requires_passive_writes_in_tool_mode
                from deerflow.agents.memory.tools import get_memory_tools

                existing_names = {tool.name for tool in extra_tools}
                for memory_tool in get_memory_tools():
                    if memory_tool.name in existing_names:
                        logger.warning("Memory tool name %r already exists and was skipped.", memory_tool.name)
                        continue
                    extra_tools.append(memory_tool)
                    existing_names.add(memory_tool.name)
                if backend_requires_passive_writes_in_tool_mode(memory_cfg.manager_class):
                    from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

                    chain.append(
                        MemoryMiddleware(
                            agent_name=name,
                            memory_config=memory_cfg,
                            pii_redaction_config=pii_redaction_config,
                        )
                    )
            else:
                if memory_cfg.mode == "tool" and not memory_cfg.enabled:
                    logger.warning("memory.mode is 'tool' but memory.enabled is false; memory tools will not be registered.")
                from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

                chain.append(
                    MemoryMiddleware(
                        agent_name=name,
                        memory_config=memory_cfg,
                        pii_redaction_config=pii_redaction_config,
                    )
                )

    # --- [10] Vision ---
    if feat.vision is not False:
        if isinstance(feat.vision, AgentMiddleware):
            chain.append(feat.vision)
        else:
            from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

            chain.append(ViewImageMiddleware())

        if feat.sandbox is not False:
            from deerflow.tools.builtins import view_image_tool

            extra_tools.append(view_image_tool)

    # --- [11] Subagent ---
    if feat.subagent is not False:
        if isinstance(feat.subagent, AgentMiddleware):
            chain.append(feat.subagent)
        else:
            from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
            from deerflow.config.subagents_config import DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN
            from deerflow.subagents.capacity import configured_subagent_max_running

            max_concurrent = subagent_runtime.config.max_running if subagent_runtime is not None else configured_subagent_max_running()
            max_total = subagent_runtime.max_total_per_run if subagent_runtime is not None else DEFAULT_MAX_TOTAL_SUBAGENTS_PER_RUN
            chain.append(
                SubagentLimitMiddleware(
                    max_concurrent=max_concurrent,
                    max_total=max_total,
                )
            )
        from deerflow.tools.builtins import task_tool

        if subagent_runtime is None:
            extra_tools.append(task_tool)
        else:
            from deerflow.tools.builtins.task_tool import bind_task_tool

            extra_tools.append(
                bind_task_tool(
                    subagent_runtime.execution_capacity,
                    app_config=subagent_runtime.app_config,
                )
            )

        if subagent_runtime is not None and subagent_runtime.batch_submitter is not None:
            from deerflow.tools.builtins.batch_task_tool import bind_batch_tools

            extra_tools.extend(
                bind_batch_tools(
                    submitter_provider=lambda: subagent_runtime.batch_submitter,
                    app_config=subagent_runtime.app_config,
                )
            )
        elif subagent_runtime is None:
            from deerflow.subagents.batch_runtime import is_subagent_batch_runtime_available

            if is_subagent_batch_runtime_available():
                from deerflow.tools.builtins import batch_status, batch_task, cancel_batch, read_batch_result

                extra_tools.extend((batch_task, batch_status, cancel_batch, read_batch_result))

    # --- [12] LoopDetection ---
    if feat.loop_detection is not False:
        if isinstance(feat.loop_detection, AgentMiddleware):
            chain.append(feat.loop_detection)
        else:
            from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
            from deerflow.config.loop_detection_config import LoopDetectionConfig

            chain.append(LoopDetectionMiddleware.from_config(LoopDetectionConfig()))

    # --- [13] TokenBudget ---
    if feat.token_budget is not False:
        if isinstance(feat.token_budget, AgentMiddleware):
            chain.append(feat.token_budget)
        else:
            from deerflow.agents.middlewares.token_budget_middleware import TokenBudgetMiddleware
            from deerflow.config.token_budget_config import TokenBudgetConfig

            # ``enabled`` defaults to False for config.yaml; ``token_budget=True`` is the opt-in.
            chain.append(TokenBudgetMiddleware.from_config(TokenBudgetConfig(enabled=True)))

    # --- [13a] ModelContentCompatibility (always) ---
    # Read-time sanitizer: rewrites URL-sourced file/image blocks in list-form
    # message content into text placeholders in the request view only, so
    # threads poisoned before the conversion-layer fix heal on the next model
    # call. Placement contract: after every built-in content transform so it
    # sees the near-final request, and before the custom/extension injection
    # point (_insert_extra) and the terminal tail — in the lead chain it is
    # followed by configured extensions, terminal-response and finish-reason
    # middlewares, and the always-last ClarificationMiddleware.
    from deerflow.agents.middlewares.model_content_compatibility_middleware import ModelContentCompatibilityMiddleware

    chain.append(ModelContentCompatibilityMiddleware())

    # --- [14] Clarification (always last among built-ins) ---
    chain.append(ClarificationMiddleware())
    extra_tools.append(ask_clarification_tool)

    # --- Insert extra_middleware via @Next/@Prev ---
    if extra_middleware:
        _insert_extra(chain, extra_middleware)
        # Invariant: ClarificationMiddleware must always be last.
        # @Next(ClarificationMiddleware) could push it off the tail.
        clar_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
        if clar_idx != len(chain) - 1:
            chain.append(chain.pop(clar_idx))

    return chain, extra_tools


# ---------------------------------------------------------------------------
# Internal: extra middleware insertion with @Next/@Prev
# ---------------------------------------------------------------------------


# ★ 额外中间件的插入算法（为什么不能直接 append）
#   调用方传入的扩展中间件常常需要「紧跟在 X 之后」或「紧贴 Y 之前」，
#   因为它们读取/改写 X/Y 建立的上下文。直接 append 到链尾会破坏这个
#   「相邻性契约」，所以这里用 @Next/@Prev 这种锚点（anchor）机制。
#   两个难点：
#   1. 冲突检测：两个扩展同时声称要贴着同一个锚点，谁的先后是错的，
#      与其瞎猜，不如报错让人显式解决。
#   2. 跨扩展锚定：A 锚在 B 上、B 又锚在 C 上，需要多轮迭代才能全部就位
#      （所以用 pending 多轮循环；若某一轮毫无进展，就说明锚点找不到或成环）。
def _insert_extra(chain: list[AgentMiddleware], extras: list[AgentMiddleware]) -> None:
    """Insert extra middlewares into *chain* using ``@Next``/``@Prev`` anchors.

    Algorithm:
      1. Validate: no middleware has both @Next and @Prev.
      2. Conflict detection: two extras targeting same anchor (same or opposite direction) → error.
      3. Insert unanchored extras before ClarificationMiddleware.
      4. Insert anchored extras iteratively (supports cross-external anchoring).
      5. If an anchor cannot be resolved after all rounds → error.
    """
    next_targets: dict[type, type] = {}
    prev_targets: dict[type, type] = {}

    anchored: list[tuple[AgentMiddleware, str, type]] = []
    unanchored: list[AgentMiddleware] = []

    for mw in extras:
        next_anchor = getattr(type(mw), "_next_anchor", None)
        prev_anchor = getattr(type(mw), "_prev_anchor", None)

        if next_anchor and prev_anchor:
            raise ValueError(f"{type(mw).__name__} cannot have both @Next and @Prev")

        if next_anchor:
            if next_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {next_targets[next_anchor].__name__} both @Next({next_anchor.__name__})")
            if next_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Next({next_anchor.__name__}) and {prev_targets[next_anchor].__name__} @Prev({next_anchor.__name__}) — use cross-anchoring between extras instead")
            next_targets[next_anchor] = type(mw)
            anchored.append((mw, "next", next_anchor))
        elif prev_anchor:
            if prev_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {prev_targets[prev_anchor].__name__} both @Prev({prev_anchor.__name__})")
            if prev_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Prev({prev_anchor.__name__}) and {next_targets[prev_anchor].__name__} @Next({prev_anchor.__name__}) — use cross-anchoring between extras instead")
            prev_targets[prev_anchor] = type(mw)
            anchored.append((mw, "prev", prev_anchor))
        else:
            unanchored.append(mw)

    # Unanchored → before ClarificationMiddleware
    clarification_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
    for mw in unanchored:
        chain.insert(clarification_idx, mw)
        clarification_idx += 1

    # Anchored → iterative insertion (supports external-to-external anchoring)
    pending = list(anchored)
    max_rounds = len(pending) + 1
    for _ in range(max_rounds):
        if not pending:
            break
        remaining = []
        for mw, direction, anchor in pending:
            idx = next(
                (i for i, m in enumerate(chain) if isinstance(m, anchor)),
                None,
            )
            if idx is None:
                remaining.append((mw, direction, anchor))
                continue
            if direction == "next":
                chain.insert(idx + 1, mw)
            else:
                chain.insert(idx, mw)
        if len(remaining) == len(pending):
            names = [type(m).__name__ for m, _, _ in remaining]
            anchor_types = {a for _, _, a in remaining}
            remaining_types = {type(m) for m, _, _ in remaining}
            circular = anchor_types & remaining_types
            if circular:
                raise ValueError(f"Circular dependency among extra middlewares: {', '.join(t.__name__ for t in circular)}")
            raise ValueError(f"Cannot resolve positions for {', '.join(names)} — anchors {', '.join(a.__name__ for _, _, a in remaining)} not found in chain")
        pending = remaining
