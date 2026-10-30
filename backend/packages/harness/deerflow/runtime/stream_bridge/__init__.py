"""Stream bridge — decouples agent workers from SSE endpoints.

A ``StreamBridge`` sits between the background task that runs an agent
(producer) and the HTTP endpoint that pushes Server-Sent Events to
the client (consumer).  This package provides an abstract protocol
(:class:`StreamBridge`) plus a default in-memory implementation backed
by :mod:`asyncio.Queue`.

【中文导读 · 初学者版】
这个目录解决的问题是：「Agent 在后台慢慢跑，浏览器怎么实时看到中间过程？」

● 生产者（producer）：真正执行 LangGraph 图的那个后台任务，它不断产生事件
  （例如「开始调工具了」「模型吐出一个字」「跑完了」）。
● 消费者（consumer）：HTTP 层，把事件转成 SSE 文本推给浏览器。

两者不能直接调函数，因为可能不在同一个协程/进程里，所以中间加一根「管道」，
这根管道就叫 StreamBridge（流式桥）。

● 涉及知识点：
  1) 生产者-消费者模式：解耦「谁产生数据」和「谁消费数据」。
  2) asyncio.Queue：Python 的异步队列，天然支持「一边塞、一边取、取不到就等」。
  3) SSE（text/event-stream）：HTTP 上「服务器持续推、客户端持续收」的技术，
     浏览器端因此能做出「打字机」效果。
  4) 哨兵值（sentinel）：END_SENTINEL / HEARTBEAT_SENTINEL 是特殊标记对象，
     用来表示「流结束了」「我还活着（心跳）」——比用 None 更安全，
     因为 None 可能本身就是合法数据。

● 换实现不改代码：本目录还有 memory（进程内，单机够用）和 redis（多进程/多机器）
  两种实现，通过 config 选择。这就是「面向接口编程」。
"""

# make_stream_bridge：读配置，决定用内存版还是 Redis 版（工厂函数）
from .async_provider import make_stream_bridge

# base：抽象协议 + 事件数据结构 + 两个哨兵值
#   StreamBridge  = 接口（抽象基类），规定有哪些方法
#   StreamEvent   = 一条事件的载体
from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent, StreamGap, StreamItem, StreamProducerLeaseState

# memory：默认实现，用 asyncio.Queue 存在本进程内存里
#   ★ 单机学习用这个就够了；多进程部署必须换 Redis，否则 A 进程产生的事件 B 进程取不到
from .memory import MemoryStreamBridge

# NOTE: ``RedisStreamBridge`` is intentionally NOT imported here. ``redis`` is an
# optional extra, and this package is pulled in transitively by ``deerflow.runtime``
# at process startup everywhere. Importing ``.redis`` eagerly would import
# ``redis.asyncio`` in every process (even memory-only/single-process ones) and
# couple every install to the redis package. It is imported lazily inside
# ``make_stream_bridge`` only when ``stream_bridge.type == "redis"``. Import it
# directly from ``deerflow.runtime.stream_bridge.redis`` if you need the class.

__all__ = [
    "END_SENTINEL",
    "HEARTBEAT_SENTINEL",
    "MemoryStreamBridge",
    "StreamBridge",
    "StreamEvent",
    "StreamGap",
    "StreamItem",
    "StreamProducerLeaseState",
    "make_stream_bridge",
]
