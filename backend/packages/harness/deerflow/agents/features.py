"""Declarative feature flags and middleware positioning for create_deerflow_agent.

Pure data classes and decorators — no I/O, no side effects.

【中文导读 · 初学者版】
这个文件是「功能开关清单」，回答一个问题：
拼装 Agent 时，要不要给它装沙箱、记忆、上下文压缩、子 Agent、护栏？

● 关键设计：声明式（declarative）
  这里只有「数据」和「装饰器」，没有任何实际动作（不读文件、不连数据库）；
  真正的装配发生在 factory.py 里。好处：一眼能看全系统能力，改行为不用改流程代码。

● 三态开关（本项目的小技巧）：
      True          → 用内置默认实现
      False         → 彻底关掉
      传中间件实例   → 用你自己写的替代实现
  类型写作 bool | AgentMiddleware，这就是「联合类型」。

● 涉及知识点：
  1) dataclass：自动生成 __init__ 的数据类，适合只装数据的结构。
  2) 装饰器（decorator）：@Next / @Prev 给类打标记，告诉装配器「我排在谁前面/后面」。
  3) 中间件顺序很重要：上下文压缩必须在「发给模型之前」发生，否则模型已经看到超长上下文了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from deerflow.config.memory_config import MemoryConfig


@dataclass
class RuntimeFeatures:
    """Declarative feature flags for ``create_deerflow_agent``.

    Most features accept:
    - ``True``: use the built-in default middleware
    - ``False``: disable
    - An ``AgentMiddleware`` instance: use this custom implementation instead

    ``summarization`` and ``guardrail`` have no built-in default — they only
    accept ``False`` (disable) or an ``AgentMiddleware`` instance (custom).
    """

    # 沙箱：工具执行时的隔离环境（防 Agent 乱删文件）。默认开。
    sandbox: bool | AgentMiddleware = True
    # 长期记忆：跨会话记住用户偏好。默认关（要额外的存储和模型调用）。
    memory: bool | AgentMiddleware = False
    # Explicit memory config for direct create_deerflow_agent(features=...) callers.
    # The lead-agent AppConfig path passes resolved_app_config.memory directly.
    memory_config: MemoryConfig | None = None
    # 上下文压缩：token 快超限时把旧消息压成摘要（没有内置默认，必须显式给）
    summarization: Literal[False] | AgentMiddleware = False
    # 子 Agent：把任务拆给下级 Agent 并行处理
    subagent: bool | AgentMiddleware = False
    # 视觉：让模型能看图片
    vision: bool | AgentMiddleware = False
    # 自动命名：根据首条消息给会话起标题
    auto_title: bool | AgentMiddleware = False
    # 护栏：拦住危险操作（例如 rm -rf）
    guardrail: Literal[False] | AgentMiddleware = False
    # 循环检测：防止 Agent 反复调同一个工具陷入死循环（保险丝）
    loop_detection: bool | AgentMiddleware = True
    # token 预算：给一次运行设上限，超了就收尾（省钱）
    token_budget: bool | AgentMiddleware = False


# ---------------------------------------------------------------------------
# Middleware positioning decorators
# ---------------------------------------------------------------------------


def Next(anchor: type[AgentMiddleware]):
    """Declare this middleware should be placed after *anchor* in the chain."""
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Next expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._next_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator


def Prev(anchor: type[AgentMiddleware]):
    """Declare this middleware should be placed before *anchor* in the chain."""
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Prev expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._prev_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator
