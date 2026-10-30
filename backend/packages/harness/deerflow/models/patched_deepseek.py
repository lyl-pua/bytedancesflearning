# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路里的位置：
#   用户提问 → Gateway 收到请求 → Agent 图节点调用大模型（ChatModel）
#   → ChatModel 把 Python 里的「消息列表」翻译成 HTTP 请求体
#   → 发给 DeepSeek / MoonShin(Kimi) 这类兼容 OpenAI 协议的推理服务。
#   本文件就站在「ChatModel 把消息翻译成 HTTP JSON」这一环。
#
# ★ 它解决什么问题：
#   DeepSeek 的「思考模式」(thinking) 会额外返回一个 reasoning_content 字段（思维链）。
#   官方 SDK / LangChain 的默认实现只把它塞进 additional_kwargs，回传历史消息时又丢掉它，
#   于是多轮对话第二轮就会报 400 错误（API 要求所有 assistant 消息都带 reasoning_content）。
#   这里通过「继承 + 覆盖方法」打补丁，把该字段补回去，同时顺手兼容 Kimi K3.1 的采样参数、
#   以及补回缓存写入 token 计数(cache_write_tokens)。
#
# ● 涉及的知识点：
#   1. 继承与「猴子补丁式」覆盖：不修改上游库源码，而是继承 ChatDeepSeek 并覆盖 protected 方法。
#   2. LangChain ChatModel 协议：模型类只需实现 _generate / _stream 等钩子，周边框架据此工作。
#   3. _get_request_payload：LangChain 内部把消息转成「请求 JSON」的地方，是最好的注入点。
#   4. reasoning_content（思维链）：模型「想的过程」，与最终回答 content 分开存放。
#   5. tool_calls / Function Calling：模型要求调用工具时返回的结构化字段。
#   6. 流式(streaming) 与 非流式 两条路径：非流式走 _create_chat_result，
#      流式走 _convert_chunk_to_generation_chunk，两处都要打补丁，否则行为不一致。

"""Patched ChatDeepSeek that preserves reasoning_content in multi-turn conversations.

This module provides a patched version of ChatDeepSeek that properly handles
reasoning_content when sending messages back to the API. The original implementation
stores reasoning_content in additional_kwargs but doesn't include it when making
subsequent API calls, which causes errors with APIs that require reasoning_content
on all assistant messages when thinking mode is enabled.
"""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_deepseek import ChatDeepSeek

from deerflow.config.managed_model_providers import MOONSHOT_K31_MODELS
from deerflow.models.assistant_payload_replay import restore_assistant_payloads, restore_reasoning_content


# ● MoonShin(Kimi) K3.1 的「特殊照顾」：只有当 base_url 是官方 https://api.moonshot.cn/v1
#   且模型名在已知清单里时才生效。这里之所以连 scheme/host/port/path 都比对一遍，
#   是为了避免把用户自建的「OpenAI 兼容代理」误判成官方服务（代理往往不支持同样的参数集）。
def _is_moonshot_k31_request(api_base: str | None, payload: dict[str, Any]) -> bool:
    """Match the official endpoint and model that the SDK will actually send."""
    extra_body = payload.get("extra_body")
    model = extra_body.get("model", payload.get("model")) if isinstance(extra_body, Mapping) else payload.get("model")
    if not isinstance(model, str) or model not in MOONSHOT_K31_MODELS:
        return False
    try:
        endpoint = urlsplit(api_base or "")
        return endpoint.scheme == "https" and endpoint.hostname == "api.moonshot.cn" and endpoint.port in (None, 443) and endpoint.path.rstrip("/") == "/v1"
    except ValueError:
        return False


# K3.1 服务端自己决定 temperature/top_p 等采样参数，客户端传了反而报错，
# 所以这里把这些键「删干净」；连 extra_body 里嵌套的同名键也要删（SDK 最后才合并 extra_body）。
# ★ 注意：这里是「就地修改」(in-place)，说明调用方期望拿到的是同一个 dict 对象。
def _normalize_moonshot_k31_payload(payload: dict[str, Any]) -> None:
    """Let K3.1 supply its fixed sampling values; it accepts no thinking switch."""
    omitted = {"thinking", "temperature", "top_p", "n", "presence_penalty", "frequency_penalty"}
    for key in omitted:
        payload.pop(key, None)
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, Mapping):
        # The SDK merges extra_body last. Copy it to preserve caller-owned values.
        payload["extra_body"] = {key: value for key, value in extra_body.items() if key not in omitted}


# 判断「这次请求到底开没开思考模式」。为什么要检查多个来源(sources)？
# 因为思考开关在不同调用层可能写在三个不同位置：构造参数 kwargs、顶层 payload、
# 以及 extra_body 里嵌套的位置。只有都看一眼，才不会漏判。
def _thinking_enabled(*sources: Any) -> bool:
    """Return whether the request explicitly enables DeepSeek thinking mode."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        thinking = source.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            return True
        extra_body = source.get("extra_body")
        if isinstance(extra_body, dict):
            nested = extra_body.get("thinking")
            if isinstance(nested, dict) and nested.get("type") == "enabled":
                return True
    return False


# ★ 本文件的核心补丁：把「历史里那条 assistant 消息」在发给服务端前补齐。
# 三个细节都来自 DeepSeek 服务的硬性要求：
#   a) 思维链必须带回去（restore_reasoning_content）；
#   b) 调用过工具的消息，content 必须是 ""，不能是 null；
#   c) 思考模式下，即使没有产生思维链，也必须给一个空字符串占位。
def _restore_deepseek_assistant_payload(
    payload_msg: dict[str, Any],
    orig_msg: AIMessage,
    *,
    thinking_enabled: bool,
) -> None:
    """Restore assistant history and required thinking-mode placeholders."""
    restore_reasoning_content(payload_msg, orig_msg)
    has_tool_calls = bool(payload_msg.get("tool_calls"))
    if has_tool_calls and payload_msg.get("content") is None:
        # DeepSeek requires an empty string, rather than null, for tool-call history.
        payload_msg["content"] = ""
    if thinking_enabled and has_tool_calls and "reasoning_content" not in payload_msg:
        # Thinking-mode tool turns require this field even when no reasoning was emitted.
        payload_msg["reasoning_content"] = ""


# ● 自定义模型类：继承官方 ChatDeepSeek，只覆盖需要改的两个 protected 钩子。
# 这样做的好处是不用 fork 整个库；只要上游升级，没被覆盖的部分自动跟着升级。
class PatchedChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek with proper reasoning_content preservation.

    When using thinking/reasoning enabled models, the API expects reasoning_content
    to be present on ALL assistant messages in multi-turn conversations. This patched
    version ensures reasoning_content from additional_kwargs is included in the
    request payload.
    """

    # LangChain 的序列化协议标记：返回 True 表示该类可被 lc_serializable 机制序列化。
    # 这是框架约定，不是因为业务需要；保持与父类一致即可。
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    # 声明「哪些属性是密钥」：LangChain 把它们从序列化结果里遮掉，避免 API Key 泄漏到
    # 日志 / trace / 前端。这里同时写了新旧两个属性名，因为上游改过字段名。
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "DEEPSEEK_API_KEY", "openai_api_key": "DEEPSEEK_API_KEY"}

    # ★ 关键覆盖点 1：拿到即将发送的 HTTP 请求体(payload)，在发送前动手补字段。
    # 调用顺序：先让父类生成「标准 payload」→ 再做自家修正 → 最后原样返回给 HTTP 层。
    # 之所以选这个钩子，是因为它是「消息已转成 JSON、但还没发出去」的唯一空窗期。
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload with reasoning_content preserved.

        Overrides the parent method to inject reasoning_content from
        additional_kwargs into assistant messages in the payload.
        """
        # 先把入参（可能是字符串/消息列表/PromptValue 等多种形态）统一转成 message 对象列表。
        original_messages = self._convert_input(input_).to_messages()
        # 过滤掉「错误兜底」消息：这类 AIMessage 只为让图能继续跑，属于内部产物，不该出现在真实请求里，
        # 否则会污染上下文、甚至让模型重复道歉。
        request_messages = [message for message in original_messages if not (isinstance(message, AIMessage) and (message.additional_kwargs or {}).get("deerflow_error_fallback"))]

        # Call parent to get the base payload
        payload = super()._get_request_payload(request_messages, stop=stop, **kwargs)

        # 先判断是不是 Kimi K3.1：如果是，就先把会被服务端拒绝的采样参数清理掉，
        # 后面的「思考开关判定」也顺带把它算成开启状态。
        moonshot_k31 = _is_moonshot_k31_request(self.api_base, payload)
        if moonshot_k31:
            _normalize_moonshot_k31_payload(payload)

        # 判断本次请求的思考模式状态：三个来源任意一个为 true 就算开启。
        request_thinking_enabled = moonshot_k31 or _thinking_enabled(
            payload,
            kwargs,
            {"extra_body": getattr(self, "extra_body", None)},
        )
        # ★ 这里是「响应→补丁」的最后一跳：按顺序把 payload 里的消息与原始消息配对，
        # 逐条调用 _restore_deepseek_assistant_payload 补 reasoning_content / 空占位。
        restore_assistant_payloads(
            payload.get("messages", []),
            request_messages,
            lambda payload_msg, orig_msg: _restore_deepseek_assistant_payload(
                payload_msg,
                orig_msg,
                thinking_enabled=request_thinking_enabled,
            ),
        )

        return payload

    # ★ 关键覆盖点 2（非流式）：把服务端返回的原始响应转成 LangChain 的 ChatResult 时，
    # 顺手补回上游转换逻辑会丢掉的 cache_write_tokens（缓存写入 token 数）。
    # 为什么这件事重要：它影响计费与用量统计，丢了会让「缓存命中率」看起来永远是 0。
    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Preserve Moonshot's ``cache_write_tokens`` in the usage metadata.

        The locked ``langchain-openai==1.2.1`` conversion reads
        ``prompt_tokens_details.cached_tokens`` (mapped to ``cache_read``) but
        not ``cache_write_tokens``, so providers such as Kimi K3.1 lose their
        cache-creation count before it reaches the usage journal (#7426).
        Restore it as ``input_token_details["cache_creation"]`` from the raw
        SDK response.
        """
        result = super()._create_chat_result(response, generation_info)

        # 即便读取失败也不能影响主流程：拿不到 token 明细就原样返回父类结果。
        if not isinstance(response, openai.BaseModel):
            return result
        prompt_details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
        cache_write_tokens = getattr(prompt_details, "cache_write_tokens", None)
        if cache_write_tokens is None:
            return result
        for generation in result.generations:
            usage = getattr(generation.message, "usage_metadata", None)
            if not usage:
                continue
            input_details = usage.setdefault("input_token_details", {})
            input_details["cache_creation"] = cache_write_tokens
        return result

    # ★ 关键覆盖点 3（流式）：流式响应是一个个 chunk 拼起来的，用量信息通常落在最后一块。
    # 与非流式补丁完全对称——只补「最后一个 usage chunk」，不碰正文内容。
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Streaming mirror of ``_create_chat_result``'s cache_write restore.

        Streamed runs (the web UI's ``messages-tuple`` mode) build usage
        metadata in ``_convert_chunk_to_generation_chunk`` ->
        ``_create_usage_metadata``, which ignores ``cache_write_tokens``
        (#7426). Restore it the same way for the final usage chunk.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None
        prompt_details = ((chunk or {}).get("usage") or {}).get("prompt_tokens_details") or {}
        cache_write_tokens = prompt_details.get("cache_write_tokens")
        if cache_write_tokens is None:
            return generation_chunk
        usage = getattr(generation_chunk.message, "usage_metadata", None)
        if not usage:
            return generation_chunk
        input_details = usage.setdefault("input_token_details", {})
        input_details["cache_creation"] = cache_write_tokens
        return generation_chunk
