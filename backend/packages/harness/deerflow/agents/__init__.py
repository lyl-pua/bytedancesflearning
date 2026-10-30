"""Agent 装配入口（这是 LangGraph 找到「主图」的地方）。

【中文导读 · 初学者版】
langgraph.json 里写着：
    "graphs": { "lead_agent": "deerflow.agents:make_lead_agent" }
翻译：图的工厂 = `deerflow.agents` 模块里的 `make_lead_agent` 函数。

所以本文件就是「Agent 世界的门」。它做两件事：
1) 把常用的类（ThreadState 等）暴露出去，让外部 import 更短；
2) 用「延迟导入」避免一启动就把所有重依赖都加载（很慢）。

● 涉及知识点：
  1) 模块级 __getattr__：Python 3.7+ 的「属性被访问时才临时生成」的钩子，
     常用来做**懒加载**（lazy import）。不 import 就不用付加载成本。
  2) LangGraph Server 用模块的 __dict__ 找工厂函数，所以入口函数必须真实存在，
     不能只靠 __getattr__ 提供。
  3) 为什么先 prime_enabled_skills_cache()：把「有哪些技能」提前读进缓存，
     免得每次对话都去扫描磁盘（性能常识）。
"""

from typing import TYPE_CHECKING, Any

from .features import Next, Prev, RuntimeFeatures

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

__all__ = [
    "create_deerflow_agent",
    "RuntimeFeatures",
    "Next",
    "Prev",
    "make_lead_agent",
    "SandboxState",
    "DeltaThreadState",
    "ThreadState",
]


def make_lead_agent(config: "RunnableConfig") -> Any:
    """Build the lead graph while keeping package-root imports lightweight.

    LangGraph Server resolves configured graph factories directly from a
    module's ``__dict__``, so this entrypoint must be a concrete module-level
    function rather than a value supplied only through ``__getattr__``.

    【中文导读】这里只是「薄薄一层转交」：
    真正搭图的代码在 agents/lead_agent/agent.py 里。这样拆是为了不被循环导入缠住。
    """
    # 延迟导入：只有真的要建图时才加载这些重模块
    from .lead_agent import make_lead_agent as factory
    from .lead_agent.prompt import prime_enabled_skills_cache

    # 预热技能缓存（避免每轮对话都扫磁盘）
    prime_enabled_skills_cache()
    return factory(config)


def __getattr__(name: str):
    """懒加载：只有当有人 `from deerflow.agents import create_deerflow_agent` 时才真的导入。"""
    if name == "create_deerflow_agent":
        from .factory import create_deerflow_agent

        globals()[name] = create_deerflow_agent  # 放进 globals，下次就不用再走这里
        return create_deerflow_agent
    if name in {"DeltaThreadState", "SandboxState", "ThreadState"}:
        from .thread_state import DeltaThreadState, SandboxState, ThreadState

        exports = {
            "DeltaThreadState": DeltaThreadState,
            "SandboxState": SandboxState,
            "ThreadState": ThreadState,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
