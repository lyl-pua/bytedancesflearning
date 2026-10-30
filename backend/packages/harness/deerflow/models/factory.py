# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路里的位置：
#   config.yaml 里写着「我有哪些模型、各用什么类、要不要开思考」
#   → 运行时按名字取一个模型 → 这里把「配置项」翻译成「构造参数」→ new 出 ChatModel 实例
#   → 交给 Agent 使用。它是「配置」到「对象」之间的转换器。
#
# ★ 它解决什么问题：
#   各家模型厂商的 SDK 参数名字都不统一（thinking / enable_thinking / extra_body.thinking …），
#   而配置文件只有一个 `use:` 字符串。这个工厂负责：
#   1. 按字符串动态 import 出模型类（可以理解成「类名写在配置里」的插件机制）；
#   2. 把配置里不属于模型构造器的元数据（名字、价格、上下文长度…）剔掉；
#   3. 把统一的「思考开关」翻译成各家方言（OpenAI extra_body / vLLM chat_template / Anthropic 原生 …）；
#   4. 顺手修掉一批「用户写错 key 就会在请求阶段炸」的坑，并给出可读的告警。
#
# ● 涉及的知识点：
#   1. 工厂模式（Factory）：调用方只给「模型名」，不关心实例怎么拼装。
#   2. 动态 import（importlib）：resolve_class 按 "包.模块:类" 字符串加载类，配置即可换实现。
#   3. LangChain ChatModel 协议：所有模型最终都满足 BaseChatModel 接口，
#      上层 Agent 只依赖接口，不依赖具体厂商。
#   4. 深合并(deep merge) 与「不要别名共享」：配置字典会被复用，
#      所以要 deepcopy，否则一次调用的修改会污染下一次。
#   5. 能力契约(contract)：把「支不支持思考 / 支持的 effort 词表」声明化，集中校验一次。
#   6. tracing 回调：Langfuse / LangSmith 通过 callbacks 注入，而不是硬编码埋点。

import copy
import logging
import os

from langchain.chat_models import BaseChatModel
from langchain_openai.chat_models.base import BaseChatOpenAI

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.models.reasoning import ReasoningContract, ReasoningPolicyError, resolve_reasoning_contract, resolve_reasoning_request
from deerflow.reflection import resolve_class
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)

# ``ModelConfig`` fields that describe the profile rather than configure the
# provider client. They are stripped before the constructor sees the profile.
# ★ 为什么需要这份「要剔除的字段清单」：ModelConfig 是 extra="allow" 的宽松模型，
# 里面既有「给构造器用的真实参数」，也有「纯粹给 UI / 调度用的元数据」。
# 如果把后者也塞给模型构造器，就会被当成未知 kwarg 转发到 HTTP 请求体里，运行期报 400。
# 用 frozenset 是因为它是常量集合，查询 O(1)，且能防止被误改。
_MODEL_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "use",
        "name",
        "display_name",
        "aliases",
        "description",
        "supports_thinking",
        "supports_reasoning_effort",
        # Declarative reasoning contract (issue #5073); consumed by the
        # normalization layer, never a provider kwarg.
        "reasoning",
        "when_thinking_enabled",
        "when_thinking_disabled",
        "thinking",
        "supports_vision",
        # Runtime/UI metadata used to size the context indicator. Provider
        # clients do not accept this as a model-constructor argument.
        "context_window",
        # Presentation-only metadata (consumed by the console's cost
        # display) — must never reach the provider client, which would
        # forward unknown kwargs into the completion request payload.
        "pricing",
        "request_admission",
    }
)


# 递归合并两个 dict：遇到嵌套 dict 就继续往下合，遇到普通值就直接覆盖。
# ★ 关键在「不修改入参」：base 先复制一份再改，避免调用方传进来的配置被悄悄改写。
def _deep_merge_dicts(base: dict | None, override: dict) -> dict:
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = dict(base or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        elif isinstance(value, dict):
            # Copy, don't alias: *override* is usually a template straight from
            # the cached ``ModelConfig`` (see ``_merge_settings``).
            merged[key] = copy.deepcopy(value)
        else:
            merged[key] = value
    return merged


# The two spellings of the vLLM/Qwen chat-template thinking switch: the legacy
# ``thinking`` alias DeerFlow documented first and ``enable_thinking``, which
# vLLM's Qwen reasoning parser reads. ``VllmChatModel`` maps the alias onto
# ``enable_thinking`` just before sending; other OpenAI-compatible classes
# forward both keys as written.
_VLLM_THINKING_SWITCHES = ("thinking", "enable_thinking")


# 关闭思考时，需要把「同一含义的两种写法」都显式置为 False；
# 只置一种的话，服务端可能仍读到另一处留下的 True，导致开关看起来没生效。
def _vllm_disable_chat_template_kwargs(chat_template_kwargs: dict) -> dict:
    """Build the disable payload for vLLM/Qwen chat template kwargs."""
    return {key: False for key in _VLLM_THINKING_SWITCHES if key in chat_template_kwargs}


# 与 _deep_merge_dicts 类似，但是「就地」把 payload 合进 settings。
# 之所以要 deepcopy，见下面中文注释：模板来自缓存的 ModelConfig，共享引用会串味。
def _merge_settings(settings: dict, payload: dict) -> None:
    """Deep-merge *payload* into the constructor settings in place.

    Nested dicts (``extra_body``, ``thinking``, ``chat_template_kwargs``) merge
    recursively so a synthesized thinking block never clobbers unrelated keys an
    operator put beside it (e.g. GLM's ``extra_body.tool_stream``).
    """
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = _deep_merge_dicts(settings[key], value)
        elif isinstance(value, dict):
            # Never hand the constructor (or a later in-place edit here) the
            # template's own mapping: payloads come straight from the cached
            # ``ModelConfig``, and a shared reference would leak one call's
            # adjustments into every later ``create_chat_model`` for that profile.
            settings[key] = copy.deepcopy(value)
        else:
            settings[key] = value


# 小工具：安全地取出 settings["extra_body"]["chat_template_kwargs"]，
# 任何一层不是 dict 就返回 None，避免后续代码到处写 isinstance 判断。
def _chat_template_kwargs(settings: dict) -> dict | None:
    """``settings["extra_body"]["chat_template_kwargs"]`` when both levels are mappings, else ``None``."""
    extra_body = settings.get("extra_body")
    if not isinstance(extra_body, dict):
        return None
    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    return chat_template_kwargs if isinstance(chat_template_kwargs, dict) else None


# 合并「思考开关」模板。★ 这里最容易出错的地方是：vLLM 的开关有两个同义拼写，
# 如果 base 写 thinking:false、模板写 enable_thinking:true，合并后两个键同时存在且值相反，
# 服务端读到哪个全看运气。所以合并之后要把值「镜像」到另一种拼写上，让两者一致。
def _merge_thinking_payload(settings: dict, payload: dict) -> None:
    """Deep-merge a thinking on/off *payload*, keeping its vLLM switch authoritative.

    Keys are never removed: a template can only add or override, so a profile's
    unrelated ``extra_body`` / ``chat_template_kwargs`` entries survive in both
    directions. The one thing a plain deep merge gets wrong is the vLLM switch,
    which has two spellings (see ``_VLLM_THINKING_SWITCHES``): a profile that
    spells it differently from the payload (base ``enable_thinking: false``,
    template ``thinking: true``) would keep both keys with opposite values, and
    whichever key the server reads might be the profile's. So after merging, the
    value the payload declares is mirrored onto the *other* spelling when the
    profile carries it. Both keys then agree: ``VllmChatModel`` collapses them to
    ``enable_thinking``, and a plain OpenAI-compatible class sends both, so the
    server sees the payload's intent under whichever name it honors. No spelling
    is invented (a template using only ``thinking`` never grows
    ``enable_thinking``), and a payload that spells both itself is left as
    written. Non-mapping template values are forwarded unchanged, as before.

    Deep-merging also means a template cannot *remove* a key by omitting it:
    a base ``extra_body.thinking.budget_tokens`` survives a disable template
    that only sets ``thinking.type: disabled``. Enable-only keys therefore
    belong in ``when_thinking_enabled``, not in the profile's base settings.
    """
    _merge_settings(settings, payload)
    payload_kwargs = _chat_template_kwargs(payload)
    merged_kwargs = _chat_template_kwargs(settings)
    if payload_kwargs is None or merged_kwargs is None:
        return
    declared = {key: payload_kwargs[key] for key in _VLLM_THINKING_SWITCHES if key in payload_kwargs}
    if len(declared) != 1:
        return
    value = next(iter(declared.values()))
    # In-place is safe: ``_merge_settings`` / ``_deep_merge_dicts`` copy every
    # template mapping they store, so ``merged_kwargs`` is never the config's own dict.
    for key in _VLLM_THINKING_SWITCHES:
        if key not in declared and key in merged_kwargs:
            merged_kwargs[key] = value


# 按「点号路径」写值，比如 reasoning_effort 或 extra_body.thinking.effort。
# 为什么要支持路径：不同厂商把 effort 放在不同层级，用统一路径写法能少一堆分支。
def _set_dotted_setting(settings: dict, path: str, value: object) -> None:
    """Write *value* at a dotted *path* (``reasoning_effort`` or ``extra_body.thinking.effort``)."""
    head, _, rest = path.partition(".")
    if not rest:
        settings[head] = value
        return
    nested: dict = {}
    cursor = nested
    parts = rest.split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    _merge_settings(settings, {head: nested})


# 「方言推断」：配置里只写了模板，但没写 dialect: xxx，那就看模板长什么样来猜厂商方言。
# 猜的顺序是「越具体的特征越先判断」，这样不会把 Anthropic 的 thinking 误判成 OpenAI 的 extra_body。
def _infer_dialect(effective_wte: dict) -> str:
    """Infer the thinking payload dialect from a ``when_thinking_enabled`` template.

    Mirrors the historical disable-path heuristics so ``dialect: auto`` behaves
    like the legacy factory did for the same template shape.
    """
    extra_body = effective_wte.get("extra_body")
    if not isinstance(extra_body, dict):
        # ``ModelConfig.when_thinking_enabled`` accepts unvalidated nested values,
        # so a truthy non-mapping ``extra_body`` reaches this heuristic; like a
        # null value it carries no payload shape to infer from.
        extra_body = {}
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type"):
        return "openai_extra_body"
    chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}
    if isinstance(chat_template_kwargs, dict) and any(key in chat_template_kwargs for key in _VLLM_THINKING_SWITCHES):
        return "vllm_chat_template"
    native_thinking = effective_wte.get("thinking")
    if isinstance(native_thinking, dict) and native_thinking.get("type"):
        return "anthropic"
    if "reasoning" in effective_wte:
        return "ollama"
    return "none"


def _vllm_toggle_chat_template_kwargs(chat_template_kwargs: dict, *, enabled: bool) -> dict:
    """Mirror whichever vLLM/Qwen switch the template declares; default to ``enable_thinking``."""
    toggled = {key: enabled for key in _VLLM_THINKING_SWITCHES if key in chat_template_kwargs}
    return toggled or {"enable_thinking": enabled}


# 生成「开/关」两种方言载荷的函数：把统一的布尔值翻译成各家要的 JSON 形状。
# 返回空 dict 表示「这个方言没有开关概念」，调用方合并时等于什么都不做。
def _dialect_payload(dialect: str, *, enabled: bool, chat_template_kwargs: dict | None = None) -> dict:
    """Constructor settings that switch thinking on or off for one payload dialect."""
    if dialect == "openai_extra_body":
        return {"extra_body": {"thinking": {"type": "enabled" if enabled else "disabled"}}}
    if dialect == "anthropic":
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if dialect == "vllm_chat_template":
        return {"extra_body": {"chat_template_kwargs": _vllm_toggle_chat_template_kwargs(chat_template_kwargs or {}, enabled=enabled)}}
    if dialect == "ollama":
        return {"reasoning": enabled}
    return {}


# ● 两条并列的思考设置路径：
#   「legacy 路径」= 老配置没有写 reasoning: 契约，走历史逻辑以保证完全不改变既有行为；
#   「contract 路径」= 新配置声明了能力契约，走声明式逻辑（可校验、可推断）。
# 之所以保留两条，是为了向后兼容：升级框架不能把线上已经跑通的老配置改坏。
def _apply_legacy_thinking_settings(
    settings: dict,
    model_config,
    *,
    thinking_enabled: bool,
    has_thinking_settings: bool,
    effective_wte: dict,
    requested_reasoning_effort: str | None,
    is_codex_model: bool,
) -> None:
    """Historical thinking/effort payload path for profiles without a ``reasoning:`` contract.

    Kept as-is so existing configurations retain their behavior (issue #5073
    acceptance criterion), including the synthesized ``reasoning_effort=minimal``
    on the OpenAI-compatible disable path that the contract path drops. The one
    deliberate change is how the operator's templates are applied: both
    ``when_thinking_enabled`` and ``when_thinking_disabled`` used to be
    ``dict.update``-ed onto the settings, so a template's ``extra_body`` replaced
    the profile's whole ``extra_body`` and silently dropped sibling keys such as
    GLM's ``tool_stream``. Both now go through ``_merge_thinking_payload`` like
    the synthesized ``extra_body`` disable payloads and the contract path: keys
    are never removed (so enable-only keys such as ``budget_tokens`` belong in
    ``when_thinking_enabled``, not in the base ``extra_body``), template values
    win on conflicts, and the template's vLLM switch stays authoritative across
    its two spellings. The native-Anthropic disable still assigns
    ``settings["thinking"]`` outright: that mapping is a tagged union keyed by
    ``type``, and nothing else lives beside it at the top level.
    """
    if requested_reasoning_effort is not None and not is_codex_model:
        settings["reasoning_effort"] = requested_reasoning_effort
    if thinking_enabled and has_thinking_settings and effective_wte:
        _merge_thinking_payload(settings, effective_wte)
    if not thinking_enabled:
        if model_config.when_thinking_disabled is not None:
            # User-provided disable settings win on conflicts; profile keys survive.
            _merge_thinking_payload(settings, model_config.when_thinking_disabled)
        elif has_thinking_settings and ((effective_wte.get("extra_body") or {}).get("thinking") or {}).get("type"):
            # OpenAI-compatible gateway: thinking is nested under extra_body
            _merge_thinking_payload(settings, {"extra_body": {"thinking": {"type": "disabled"}}})
            settings["reasoning_effort"] = "minimal"
        elif has_thinking_settings and (disable_chat_template_kwargs := _vllm_disable_chat_template_kwargs((effective_wte.get("extra_body") or {}).get("chat_template_kwargs") or {})):
            # vLLM uses chat template kwargs to switch thinking on/off.
            _merge_thinking_payload(settings, {"extra_body": {"chat_template_kwargs": disable_chat_template_kwargs}})
        elif has_thinking_settings and (effective_wte.get("thinking") or {}).get("type"):
            # Native langchain_anthropic: thinking is a direct constructor parameter
            settings["thinking"] = {"type": "disabled"}


# 新路径。这个函数接收的 thinking_enabled / requested_reasoning_effort 都是
# 「已经过校验的最终值」，所以这里不再需要判断「这个模型到底支不支持」。
def _apply_contract_thinking_settings(
    settings: dict,
    contract: ReasoningContract,
    model_config,
    *,
    thinking_enabled: bool,
    effective_wte: dict,
    requested_reasoning_effort: str | None,
    codex_handles_generic_effort: bool,
) -> None:
    """Thinking/effort payload path for profiles that declare a ``reasoning:`` contract.

    ``thinking_enabled`` and ``requested_reasoning_effort`` are the *resolved*
    values from :func:`resolve_reasoning_request`, so a required-thinking model
    never enters the disable branch and the effort value is already one the
    provider accepts.

    * Enabled: the dialect's enable payload deep-merged under the operator's
      ``when_thinking_enabled`` / ``thinking`` template (template wins), plus
      ``clear_thinking`` when the contract states a history requirement and the
      dialect nests thinking under ``extra_body``.
    * Disabled: ``when_thinking_disabled`` when present, else the dialect's
      disable payload. No effort value is ever synthesized here.
    * Effort: written last at ``effort.path`` so a validated request wins over
      a template value; ``None`` leaves any profile/template value in place. A
      model without an effort contract never forwards ``reasoning_effort``.
    """
    dialect = contract.dialect if contract.dialect != "auto" else _infer_dialect(effective_wte)
    # Both directions mirror whichever vLLM switch the template declares, so a
    # template using the older ``thinking`` key never also grows ``enable_thinking``.
    # ``when_thinking_enabled`` holds unvalidated nested values: a truthy
    # non-mapping ``extra_body`` has no ``chat_template_kwargs`` to mirror and is
    # treated as absent, like the null values the ``or {}`` defaults already cover.
    extra_body = effective_wte.get("extra_body")
    chat_template_kwargs = (extra_body.get("chat_template_kwargs") or {}) if isinstance(extra_body, dict) else {}
    if thinking_enabled:
        payload = _deep_merge_dicts(_dialect_payload(dialect, enabled=True, chat_template_kwargs=chat_template_kwargs), effective_wte)
        if contract.history is not None and dialect == "openai_extra_body":
            payload = _deep_merge_dicts(payload, {"extra_body": {"thinking": {"clear_thinking": contract.history == "clear"}}})
        _merge_thinking_payload(settings, payload)
    elif model_config.when_thinking_disabled is not None:
        _merge_thinking_payload(settings, model_config.when_thinking_disabled)
    else:
        _merge_thinking_payload(settings, _dialect_payload(dialect, enabled=False, chat_template_kwargs=chat_template_kwargs))
    if contract.effort is None or contract.effort.path != "reasoning_effort":
        # A custom path is the only effort wire format for this contract.
        # Drop a generic key supplied by model_overrides as well as stale
        # settings on a ModelConfig mutated after load-time validation.
        settings.pop("reasoning_effort", None)
    if contract.effort is not None and requested_reasoning_effort is not None and not codex_handles_generic_effort:
        _set_dotted_setting(settings, contract.effort.path, requested_reasoning_effort)


# 判断某个类是否「自己声明了 api_base 字段」。为什么这件事重要：
# ChatDeepSeek 这类类把 api_base 当作正规参数，改名成 base_url 反而会失效；
# 而其它 OpenAI 兼容类并不认识 api_base，需要改名。两者不能一刀切。
def _declares_api_base(model_class: type) -> bool:
    """Whether *model_class* declares ``api_base`` as its own constructor field.

    ``langchain_deepseek:ChatDeepSeek`` (and therefore ``PatchedChatDeepSeek``) does, so for it
    ``api_base`` is the canonical endpoint key and must be passed through untouched. Every other
    ``BaseChatOpenAI`` subclass inherits only ``openai_api_base`` (alias ``base_url``).
    """
    return "api_base" in getattr(model_class, "model_fields", {})


# ★ 一个非常典型的「用户写错 key」的兜底：api_base 是别家类的正规参数，
# 用户拷贝示例配置过来时很容易把它写到 OpenAI 兼容类上。这里自动改名成 base_url，
# 让配置「按用户的本意」生效，而不是等到发请求时才报一个看不懂的关键字错误。
def _normalize_openai_base_url(model_class: type, model_settings_from_config: dict) -> None:
    """Map the common ``api_base`` alias to ``base_url`` for OpenAI-compatible clients.

    ``BaseChatOpenAI`` subclasses accept the OpenAI endpoint override as ``base_url`` (with
    ``openai_api_base`` as a legacy alias). Several providers in ``config.example.yaml`` use
    ``api_base`` for *other* model classes, so users frequently copy ``api_base`` onto such a model
    by mistake. Because ``ModelConfig`` is ``extra="allow"``, the bad key is not caught at
    config-load time — it is forwarded to the constructor, which does not reject it but transfers it
    into ``model_kwargs``; that is then spread into every ``Completions.create()`` call and rejected
    by the OpenAI SDK at *request* time with an opaque ``unexpected keyword argument 'api_base'``
    error (and the endpoint override is silently dropped). Rename it here so the model works as the
    user intended.

    Gated on ``issubclass(model_class, BaseChatOpenAI)`` rather than a class-path allowlist, so any
    OpenAI-compatible subclass is covered automatically — the divert-and-crash behaviour is a
    property of the base class, not of the two paths that used to be listed. Classes that declare
    ``api_base`` themselves are skipped: there the key is canonical, not a typo.
    """
    if not issubclass(model_class, BaseChatOpenAI) or _declares_api_base(model_class):
        return
    if "api_base" not in model_settings_from_config:
        return
    if "base_url" in model_settings_from_config or "openai_api_base" in model_settings_from_config:
        # Canonical key already present; drop the alias to avoid a duplicate-intent kwarg.
        model_settings_from_config.pop("api_base", None)
        logger.warning("Model config sets both an endpoint key (base_url/openai_api_base) and 'api_base'; using the former and ignoring 'api_base'.")
        return
    model_settings_from_config["base_url"] = model_settings_from_config.pop("api_base")
    logger.debug("Normalized model config key 'api_base' -> 'base_url' for OpenAI-compatible client.")


# 「把隐患提前变成一条能看懂的日志」：未知 key 在 LangChain 里不会被拒绝，
# 而是被塞进 model_kwargs，最后在真正调用 HTTP 时才报错——排查成本极高。
# 这里在「建模型的瞬间」就把它标出来，属于 best-effort（尽力而为），绝不阻断启动。
def _warn_unknown_model_settings(model_class, model_name: str, model_settings_from_config: dict) -> None:
    """Warn about config keys the OpenAI client will silently divert into ``model_kwargs``.

    ``ModelConfig`` is ``extra="allow"``, so a typo'd key (e.g. ``maxx_tokens``) is not caught at
    config-load time. LangChain's OpenAI client does not reject an unknown constructor kwarg — it
    emits a ``UserWarning`` and transfers the key into ``model_kwargs``, which is then spread into
    every ``Completions.create()`` call and rejected by the OpenAI SDK at *request* time with an
    opaque ``unexpected keyword argument`` error that is very hard to trace back to a config typo.

    This turns that latent failure into an explicit, actionable log line at model-build time. It is
    **scoped to the OpenAI-compatible family** — that is where the ``model_kwargs``
    divert-and-crash behavior occurs and where the known field/alias set is accurate. The family is
    ``issubclass(model_class, BaseChatOpenAI)``: the divert is implemented in that base class, so
    every subclass inherits it. Other providers (e.g. ``ChatAnthropic``) route extra kwargs
    differently and would false-positive against this allow-list, so they are intentionally left
    alone. Best-effort and non-fatal: it only fires when the class exposes a pydantic
    ``model_fields`` schema, treats both field names and their aliases as valid, and allow-lists the
    standard passthrough kwargs the factory injects and the OpenAI client accepts.
    """
    if not issubclass(model_class, BaseChatOpenAI):
        return
    known = getattr(model_class, "model_fields", None)
    if not known:
        return
    valid_names = set(known.keys())
    for field in known.values():
        alias = getattr(field, "alias", None)
        if alias:
            valid_names.add(alias)
    # Standard kwargs the factory injects or the OpenAI client accepts beyond declared fields.
    valid_names |= {
        "model",
        "model_kwargs",
        "extra_body",
        "default_headers",
        "default_query",
        "stream_usage",
        "stream_chunk_timeout",
        "reasoning_effort",
    }
    unknown = sorted(k for k in model_settings_from_config if k not in valid_names)
    if unknown:
        logger.warning(
            "Model '%s' (%s): config key(s) %s are not recognized parameters of the model class and will be forwarded as-is; this may raise at request time. Check for typos (e.g. 'maxx_tokens' -> 'max_tokens').",
            model_name,
            getattr(model_class, "__name__", "?"),
            unknown,
        )


# Default chunk-gap budget for OpenAI-compatible streaming responses.
#
# langchain-openai raises ``StreamChunkTimeoutError`` after this many seconds
# without receiving a chunk. Its own default is 120s, which is too aggressive for
# reasoning models (DeepSeek-R1, Doubao-thinking, GPT-5) whose first chunk can
# legitimately take 90~150s. We default to 240s so the streaming layer rarely
# trips on long thinking pauses; the LLMErrorHandlingMiddleware still retries
# (budget=2) if a real stall happens. Users can override per-model in config.yaml.
# ★ 为什么默认是 240 秒而不是库自带的 120 秒：推理模型「第一个 chunk」经常要 90~150 秒
# 才吐出来，120 秒会把正常的长思考误判成卡死，然后重试、浪费钱。
_DEFAULT_STREAM_CHUNK_TIMEOUT_SECONDS: float = 240.0


# 给 OpenAI 兼容客户端注入合理的流式超时；非兼容客户端则把这个 key 删掉，
# 因为对它们来说这是「不认识的参数」，留着只会变成隐患。
def _apply_stream_chunk_timeout_default(model_class: type, model_settings_from_config: dict) -> None:
    """Inject a generous ``stream_chunk_timeout`` for OpenAI-compatible clients.

    ``stream_chunk_timeout`` is a field of langchain-openai's ``BaseChatOpenAI``, so
    it is accepted by ``ChatOpenAI`` and by every DeerFlow provider that subclasses
    it: ``PatchedChatOpenAI`` plus the self-hosted / reasoning adapters
    ``VllmChatModel``, ``MindIEChatModel``, ``PatchedChatDeepSeek``,
    ``PatchedChatMiMo``, ``PatchedChatStepFun`` and ``PatchedChatMiniMax``. We gate on
    ``issubclass(model_class, BaseChatOpenAI)`` rather than an explicit class-path
    allowlist so any OpenAI-compatible subclass inherits the default (and honors an
    explicit override) automatically. Issue #3189 was reported against ``mimo-v2.5``
    (``PatchedChatMiMo``); the original fix (#3195) matched only ``ChatOpenAI`` /
    ``PatchedChatOpenAI``, so those subclasses kept langchain-openai's aggressive
    built-in chunk-gap timeout and — worse — silently discarded a user's explicit
    ``stream_chunk_timeout``.

    Behaviour:

    * ``BaseChatOpenAI`` subclass: an explicit value in ``config.yaml`` is preserved.
      An explicit ``null`` is dropped upstream by ``model_dump(exclude_none=True)``
      and therefore treated as "unset", so the default is injected.
    * Any other client (e.g. ``ChatAnthropic``): drop the key so it is never
      forwarded to a constructor that does not declare it. The kwarg is not a
      declared field of these clients: depending on the client it is either
      silently dropped (``ChatAnthropic`` declares ``extra="ignore"``) or, for
      other OpenAI-style clients, diverted into ``model_kwargs`` and rejected
      at request time. Either way the user's intent is lost, so we drop it
      proactively instead.
    """
    if not issubclass(model_class, BaseChatOpenAI):
        model_settings_from_config.pop("stream_chunk_timeout", None)
        return
    if "stream_chunk_timeout" in model_settings_from_config:
        return
    model_settings_from_config["stream_chunk_timeout"] = _DEFAULT_STREAM_CHUNK_TIMEOUT_SECONDS


# gpt-6 family must use the OpenAI Responses API when tools are attached.
#
# The locked ``langchain-openai==1.2.1`` still routes tool use through
# ``/v1/chat/completions`` for every model that is not in its hard-coded
# ``_RESPONSES_API_ONLY_PREFIXES`` tuple (gpt-5-pro style names only). OpenAI
# rejects function tools for the gpt-6 family on that endpoint ("Function tools
# with reasoning_effort are not supported for gpt-6.<x> in /v1/chat/completions"),
# so a gpt-6 model configured with ``langchain_openai:ChatOpenAI`` plus any tool
# fails every tool call with a 400. Newer langchain-openai (1.7.1+) added a
# ``gpt-6 + tools`` → Responses API branch; backport that behaviour here so the
# pinned dependency does not silently break the whole gpt-6 family.
#
# The model factory creates the client before any tool is bound, so a
# "gpt-6 + tools" check is not expressible at this point — the injection is
# necessarily broader than upstream's bind-time routing. Two gates keep that
# blast radius small:
#
# * **Custom endpoints are never touched.** ``_normalize_openai_base_url`` ran
#   just above, so an OpenAI-compatible gateway/proxy configured via
#   ``base_url``/``openai_api_base`` already lands in ``model_settings_from_config``;
#   those deployments (LiteLLM, vLLM, one-api, …) often lack ``/responses`` and
#   must not be silently re-pointed. ``AzureChatOpenAI`` is skipped too: it
#   subclasses ``BaseChatOpenAI``, and Azure's Responses API additionally
#   requires ``api-version >= 2025-04-01-preview``. A gateway wired purely via
#   the ``OPENAI_BASE_URL``/``OPENAI_API_BASE`` environment variables (the openai
#   SDK's fallback when no ``base_url`` is passed) is also treated as a custom
#   endpoint. Known blind spot: a ``BaseChatOpenAI`` subclass that declares its
#   own ``api_base`` field (ChatDeepSeek-style) skips
#   ``_normalize_openai_base_url``, so its ``api_base`` never becomes ``base_url``
#   and is invisible to this gate.
# * **The injection is logged**, so an operator can trace why their model began
#   calling ``/responses`` even before they diagnose a failure.
#
# Behaviour:
#
# * An explicit ``use_responses_api`` always wins (true or false) — it is the
#   operator's routing decision, even on a custom endpoint or Azure.
# * Otherwise a ``BaseChatOpenAI`` subclass whose ``model`` starts with ``gpt-6``
#   gets ``use_responses_api: true`` + ``output_version: responses/v1`` injected,
#   unless the model targets a custom endpoint (``base_url``/``openai_api_base``,
#   or a set ``OPENAI_BASE_URL``/``OPENAI_API_BASE`` env var) or is an
#   ``AzureChatOpenAI``.
# * An explicit ``output_version`` alone (without ``use_responses_api``) implies
#   the operator expects Responses formatting: their value is preserved and
#   ``use_responses_api: true`` is still injected, so the issue #6841 400s do not
#   silently reappear.
# * Any other model family or client (e.g. ChatAnthropic) is left untouched.
# gpt-6 系列带工具时必须走 /responses（Responses API），否则每次工具调用都 400。
# 这是给锁定的 langchain-openai 版本打的「前向兼容补丁」，逻辑详见上面英文长注释。
def _apply_openai_responses_api_default(model_class: type, model_settings_from_config: dict) -> None:
    if not issubclass(model_class, BaseChatOpenAI):
        return
    model_name = model_settings_from_config.get("model")
    if not isinstance(model_name, str) or not model_name.lower().startswith("gpt-6"):
        return
    # An explicit routing choice always wins, even on a custom endpoint or Azure.
    if "use_responses_api" in model_settings_from_config:
        return
    # A custom endpoint means the operator already owns routing: OpenAI-compatible
    # gateways/proxies (LiteLLM, vLLM, one-api) configured via base_url /
    # openai_api_base (canonicalized just above by _normalize_openai_base_url),
    # and Azure deployments, rarely implement /responses (Azure's additionally
    # needs api-version >= 2025-04-01-preview). Leave them untouched. The Azure
    # check walks the MRO so it also covers env-configured Azure deployments
    # (which leave azure_endpoint out of the model settings) and any subclass.
    if any(getattr(base, "__name__", "") == "AzureChatOpenAI" for base in model_class.__mro__):
        return
    if any(key in model_settings_from_config for key in ("base_url", "openai_api_base", "azure_endpoint")):
        return
    # A gateway can also be wired purely through the environment: the openai SDK
    # falls back to OPENAI_BASE_URL/OPENAI_API_BASE when no base_url is passed,
    # so a gpt-6 model behind an env-var-wired gateway would otherwise be
    # re-pointed at <gateway>/responses — the exact regression the endpoint gate
    # exists to avoid. Treat a set env var like an explicit endpoint.
    if os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"):
        return
    if "output_version" in model_settings_from_config:
        # Operator pinned an output version — implying Responses formatting —
        # without enabling routing. Honor the value but still turn the default on,
        # otherwise gpt-6 stays on /v1/chat/completions and re-exposes #6841.
        model_settings_from_config["use_responses_api"] = True
        logger.debug(
            "gpt-6 model '%s': enabling OpenAI Responses API routing (use_responses_api=true) with operator output_version=%r",
            model_name,
            model_settings_from_config["output_version"],
        )
        return
    model_settings_from_config["use_responses_api"] = True
    model_settings_from_config["output_version"] = "responses/v1"
    logger.info(
        "gpt-6 model '%s': routing through the OpenAI Responses API (use_responses_api=true, output_version=responses/v1) because the pinned langchain-openai cannot call gpt-6 tools on /v1/chat/completions",
        model_name,
    )


# ★★★ 工厂主函数：调用方只需要「模型名 + 要不要开思考」，其余全部在这里拼好。
# 阅读顺序建议：
#   1) 找配置 → 2) 动态 import 类 → 3) 剔除元数据 → 4) 校验能力契约
#   → 5) 生成思考/effort 参数 → 6) 修 key(api_base/超时/responses) → 7) 实例化 → 8) 挂 tracing
def create_chat_model(name: str | None = None, thinking_enabled: bool = False, *, app_config: AppConfig | None = None, attach_tracing: bool = True, model_overrides: dict | None = None, **kwargs) -> BaseChatModel:
    """Create a chat model instance from the config.

    Args:
        name: The name of the model to create. If None, the first model in the config will be used.
        thinking_enabled: Enable the model's extended-thinking mode when supported.
        app_config: Explicit application config; falls back to the cached global if omitted.
        model_overrides: Optional per-caller sampling overrides (e.g. a custom
            agent's ``temperature`` / ``max_tokens``) layered on top of the
            model profile. ``None`` values are ignored so an unset override
            never clobbers a profile value. Applied before the thinking / Codex
            transforms so provider-specific normalization (e.g. Codex dropping
            ``max_tokens``) still governs an overridden value.
        attach_tracing: When True (default), attach tracing callbacks (Langfuse,
            LangSmith) directly to the model instance. Standalone callers — anything
            that invokes the model outside a LangGraph run that already wires tracing
            at the invocation root (``MemoryUpdater``, ad-hoc utilities, etc.) — keep
            this default so the model-level callback still produces traces. Callers
            that already attach tracing at the graph root (``make_lead_agent``, the
            in-graph ``TitleMiddleware``) MUST pass ``attach_tracing=False``; otherwise
            the same LLM call emits duplicate spans (one rooted at the graph, one at
            the model) and ``session_id`` / ``user_id`` metadata never reach the trace
            because the model becomes a nested observation whose ``langfuse_*`` keys
            get stripped.

    Returns:
        A chat model instance.
    """
    config = app_config or get_app_config()
    # 不指定名字就用第一个模型，这是「默认模型」的约定。
    if name is None:
        name = config.models[0].name
    model_config = config.get_model_config(name)
    if model_config is None:
        raise ValueError(f"Model {name} not found in config") from None
    # ★ 动态 import：把配置里的 "包.模块:类名" 解析成真正的类，并校验是 BaseChatModel 子类。
    model_class = resolve_class(model_config.use, BaseChatModel)
    # 转成 dict 时顺手剔掉元数据字段（exclude=...），这样下游拿到的就是「纯构造参数」。
    model_settings_from_config = model_config.model_dump(exclude_none=True, exclude=set(_MODEL_METADATA_FIELDS))
    if isinstance(model_config.reasoning, bool | str):
        # Before the capability contract, ``reasoning`` was an unrestricted
        # provider kwarg. ChatOllama uses the boolean form and, for gpt-oss
        # style models, the ``low|medium|high`` level string; keep forwarding
        # both on the legacy path even though contract metadata is excluded
        # above.
        model_settings_from_config["reasoning"] = model_config.reasoning
    # Layer per-caller sampling overrides (e.g. a custom agent's temperature /
    # max_tokens) on top of the profile. Ignore None so an unset override never
    # clobbers a configured profile value. Applied here — before the thinking
    # and Codex transforms below — so provider-specific normalization (Codex
    # dropping max_tokens, thinking disable-paths) still governs the merged
    # value exactly as it would a profile-native one.
    if model_overrides:
        model_settings_from_config.update({key: value for key, value in model_overrides.items() if value is not None})
    # The per-request reasoning effort layers the same way. The regular lead-agent
    # build forwards the key even when None (neither the request nor the custom
    # agent chose one), so it must leave kwargs: a profile that also yields
    # reasoning_effort would otherwise hand the constructor the keyword twice.
    # Codex validates and maps the requested value itself below.
    # 提前 import 只是为了做 issubclass 判断（Codex 适配器有特殊规则）。
    from deerflow.models.openai_codex_provider import CodexChatModel

    is_codex_model = issubclass(model_class, CodexChatModel)
    # 用 pop 取：同一个 key 后面还要传给构造器，留在 kwargs 里会导致「同名参数传两次」的 TypeError。
    requested_reasoning_effort = kwargs.pop("reasoning_effort", None)
    # Compute effective when_thinking_enabled by merging in the `thinking` shortcut field.
    # The `thinking` shortcut is equivalent to setting when_thinking_enabled["thinking"].
    has_thinking_settings = (model_config.when_thinking_enabled is not None) or (model_config.thinking is not None)
    effective_wte: dict = dict(model_config.when_thinking_enabled) if model_config.when_thinking_enabled else {}
    if model_config.thinking is not None:
        merged_thinking = {**(effective_wte.get("thinking") or {}), **model_config.thinking}
        effective_wte = {**effective_wte, "thinking": merged_thinking}

    # One normalized reasoning policy for every caller (issue #5073): the lead
    # agent, subagents, summarization, title generation and one-shot utilities
    # all land here, so required thinking, unsupported thinking and restricted
    # effort vocabularies are enforced in exactly one place.
    # ★ 能力契约：先把「这个模型支持什么」解析成一个对象，后面所有校验只依赖它，
    # 避免「东一处 if 支持思考、西一处 if 支持 effort」这种分散判断带来的不一致。
    contract = resolve_reasoning_contract(model_config)
    # Codex's legacy/default-path adapter writes the generic constructor key.
    # A declared custom path takes precedence even for a Codex subclass.
    codex_handles_generic_effort = is_codex_model and (contract.source == "legacy" or contract.effort is None or contract.effort.path == "reasoning_effort")
    if contract.source == "legacy" and thinking_enabled and has_thinking_settings and not contract.supports_thinking:
        raise ValueError(f"Model {name} does not support thinking. Set `supports_thinking` to true in the `config.yaml` to enable thinking.") from None
    # 归一化：把「用户想要的」与「模型能给的」取交集，得到本次真正生效的值；
    # 若用户的要求超出模型能力，就在这一处抛 ValueError，报错信息也统一好读。
    try:
        resolved_reasoning = resolve_reasoning_request(contract, thinking_enabled=thinking_enabled, reasoning_effort=requested_reasoning_effort)
    except ReasoningPolicyError as exc:
        raise ValueError(f"Model {name}: {exc}") from exc
    if resolved_reasoning.adjustments:
        logger.info("Model '%s': reasoning request adjusted by its capability contract (%s)", name, ", ".join(resolved_reasoning.adjustments))
    requested_reasoning_effort = resolved_reasoning.reasoning_effort
    if contract.source == "legacy":
        # Profiles without a ``reasoning:`` block keep the historical payload
        # path, driven by the caller's raw thinking flag; the only change from
        # the pre-contract factory is that its templates deep-merge (see the
        # function docstring) instead of replacing whole settings.
        _apply_legacy_thinking_settings(
            model_settings_from_config,
            model_config,
            thinking_enabled=thinking_enabled,
            has_thinking_settings=has_thinking_settings,
            effective_wte=effective_wte,
            requested_reasoning_effort=requested_reasoning_effort,
            is_codex_model=is_codex_model,
        )
        if not contract.supports_reasoning_effort:
            requested_reasoning_effort = None
            model_settings_from_config.pop("reasoning_effort", None)
    else:
        thinking_enabled = resolved_reasoning.thinking_enabled
        _apply_contract_thinking_settings(
            model_settings_from_config,
            contract,
            model_config,
            thinking_enabled=thinking_enabled,
            effective_wte=effective_wte,
            requested_reasoning_effort=requested_reasoning_effort,
            codex_handles_generic_effort=codex_handles_generic_effort,
        )

    # Normalize the api_base -> base_url alias FIRST, so the downstream OpenAI-compatible
    # heuristics (stream_usage default below / stream_chunk_timeout) see the canonical endpoint key.
    # ★ 顺序很重要：先归一化 endpoint key，后面的「流式超时 / Responses API」启发式
    # 才能看到统一的 base_url，否则会误判成「用户没配自定义 endpoint」。
    _normalize_openai_base_url(model_class, model_settings_from_config)
    _apply_stream_chunk_timeout_default(model_class, model_settings_from_config)
    # gpt-6 + tools must route through the Responses API (see the helper docstring).
    _apply_openai_responses_api_default(model_class, model_settings_from_config)

    # For Codex Responses API models: map thinking mode to reasoning_effort
    if is_codex_model:
        # The ChatGPT Codex endpoint currently rejects max_tokens/max_output_tokens.
        model_settings_from_config.pop("max_tokens", None)

        if codex_handles_generic_effort:
            # Use explicit reasoning_effort from frontend if provided (low/medium/high)
            if not thinking_enabled:
                model_settings_from_config["reasoning_effort"] = "none"
            elif requested_reasoning_effort in ("low", "medium", "high", "xhigh"):
                model_settings_from_config["reasoning_effort"] = requested_reasoning_effort
            elif "reasoning_effort" not in model_settings_from_config:
                model_settings_from_config["reasoning_effort"] = "medium"

    # For MindIE models: enforce conservative retry defaults.
    # Timeout normalization is handled inside MindIEChatModel itself.
    if getattr(model_class, "__name__", "") == "MindIEChatModel":
        # Enforce max_retries constraint to prevent cascading timeouts.
        model_settings_from_config["max_retries"] = model_settings_from_config.get("max_retries", 1)

    # Ensure stream_usage is enabled so that token usage metadata is available
    # in streaming responses.  LangChain's BaseChatOpenAI only defaults
    # stream_usage=True when no custom base_url/api_base is set, so models
    # hitting third-party endpoints (e.g. doubao, deepseek) silently lose
    # usage data.  We default it to True unless explicitly configured.
    # 第三方 endpoint（豆包、DeepSeek 等）默认不带 usage 统计，这里统一切成开启，
    # 否则前端看到的 token 用量会一直是 0。只有框架支持该字段时才设置，避免传错参数。
    if "stream_usage" not in model_settings_from_config and "stream_usage" not in kwargs:
        if "stream_usage" in getattr(model_class, "model_fields", {}):
            model_settings_from_config["stream_usage"] = True

    # Translate the declared context window into the langchain profile so
    # profile-dependent features (e.g. SummarizationMiddleware fraction triggers,
    # which resolve thresholds from profile["max_input_tokens"]) work for
    # third-party OpenAI-compatible models whose SDK ships no profile of its
    # own (#3103). ``profile`` is a metadata-only BaseChatModel field
    # (exclude=True) and never reaches the provider request payload. An
    # explicit profile from a caller or model_overrides is never clobbered.
    # 只有当「确实配了上下文长度」且「调用方没自己传 profile」时才翻译，
    # 避免覆盖调用方显式给出的 profile（那是更精确的信息）。
    translate_context_window = bool(model_config.context_window) and "profile" not in kwargs and "profile" not in model_settings_from_config

    if model_config.request_admission is not None:
        from deerflow.models.request_admission import get_request_admission

        if "rate_limiter" in kwargs or "rate_limiter" in model_settings_from_config:
            raise ValueError("request_admission cannot be combined with a custom rate_limiter")
        model_settings_from_config["rate_limiter"] = get_request_admission(name, model_config.request_admission)
        # SDK-internal retries do not re-enter BaseChatModel's admission hook.
        # Keep retries at the middleware layer where each attempt is paced.
        if "max_retries" in model_class.model_fields:
            kwargs.pop("max_retries", None)
            model_settings_from_config["max_retries"] = 0

    _warn_unknown_model_settings(model_class, name, model_settings_from_config)

    # 配置提供默认值，调用方显式传入的非空参数统一覆盖配置。
    # 先合并再展开，避免同名字段通过两个 **dict 传入时触发 TypeError。
    effective_model_settings = dict(model_settings_from_config)
    effective_model_settings.update({key: value for key, value in kwargs.items() if value is not None})
    model_instance = model_class(**effective_model_settings)

    if translate_context_window:
        # Applied *after* construction and merged into the provider's inferred
        # profile: passing ``profile`` to the constructor would REPLACE the whole
        # inferred metadata (tool_calling, structured_output, io capabilities,
        # output limits) with just this one key. The declared window wins on
        # max_input_tokens itself — the operator set it precisely because the
        # inferred (or absent) value doesn't match their gateway.
        # ★ 注意这里是「构造之后再改 profile」而不是构造时传参：
        # 直接传 profile= 会整个替换掉框架自动推断的能力元数据（tool_calling、输出上限…），
        # 正确做法是先取推断值，再把 max_input_tokens 覆盖进去。
        inferred_profile = getattr(model_instance, "profile", None)
        model_instance.profile = {**(inferred_profile or {}), "max_input_tokens": model_config.context_window}

    # tracing 用「挂回调」的方式接入，而不是侵入业务代码。
    # 注意：如果上层 LangGraph 已经在图级别挂了 tracing，这里必须传 attach_tracing=False，
    # 否则同一次调用会产生两条嵌套的 trace，元数据也会丢。
    if attach_tracing:
        callbacks = build_tracing_callbacks()
        if callbacks:
            existing_callbacks = model_instance.callbacks or []
            model_instance.callbacks = [*existing_callbacks, *callbacks]
            logger.debug(f"Tracing attached to model '{name}' with providers={len(callbacks)}")
    return model_instance
