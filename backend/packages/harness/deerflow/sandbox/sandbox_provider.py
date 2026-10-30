# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路里的位置：
#   Agent 决定「要跑一段代码 / 执行 shell 命令」→ 调用 bash/python 工具
#   → 工具需要一个「隔离的容器」来执行 → 于是向 SandboxProvider 申请沙箱。
#   本文件就是「沙箱生命周期接口 + 单例注册表」这一层：它不负责真正开容器，
#   只规定「谁都得实现 acquire/get/release」这套契约，并管理全局唯一实例。
#
# ★ 它解决什么问题：
#   1. 用抽象基类定义契约，让本地 Docker、云端 provisioner、AioSandbox 等多种实现可互换；
#   2. 用「单例 + 线程锁」保证整个进程只有一个 provider，避免重复初始化、重复建连；
#   3. 提供异步包装，避免阻塞操作卡死 asyncio 事件循环。
#
# ● 涉及的知识点：
#   1. ABC / @abstractmethod：抽象基类，子类必须实现才能实例化（接口约束）。
#   2. 工厂 + 单例模式：resolve_class 按配置里的字符串动态选出实现类（工厂），
#      并全局缓存一个实例（单例）。
#   3. 动态 import（importlib）：配置写的是 "包.模块:类名" 这种字符串，
#      运行期才真正 import，所以配置可以换实现而不用改代码。resolve_class 就是做这件事。
#   4. threading.Lock 与「双重检查」：多线程下安全地创建单例；锁只保护「引用替换」，
#      耗时操作（构造/清理回调）放到锁外，防止自锁(deadlock)。
#   5. asyncio.to_thread：把阻塞调用丢到线程池，等价于「别堵住事件循环」。
#   6. 沙箱隔离：Agent 生成的代码在受限环境执行，避免动到宿主机文件/进程。

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from deerflow.config import get_app_config
from deerflow.reflection import resolve_class
from deerflow.sandbox.lease import run_sync_lifecycle_operation
from deerflow.sandbox.sandbox import Sandbox

if TYPE_CHECKING:
    from deerflow.skills.projection import SkillProjectionPaths


# ● 抽象基类：只定义「接口形状」，不提供实现。
# 注意这里几个类属性不是配置项，而是「能力声明」——上层代码据此决定能不能用某项功能，
# 所以子类可以覆盖它们来告诉外面「我支持 / 我不支持」。
class SandboxProvider(ABC):
    """Abstract base class for sandbox providers"""

    # 是否需要把线程目录以 bind mount 方式挂进沙箱（本地 Docker 是 True，云端上传式是 False）。
    uses_thread_data_mounts: bool = False
    # 是否需要在挂载后调整上传文件权限（上传型 provider 通常要，本地挂载不需要）。
    needs_upload_permission_adjustment: bool = True
    # Capability for enforcing a lead Agent's physical skill view across the
    # provider's current Agent-accessible tool surface. Host-backed providers
    # must return False whenever shell access can bypass managed path mappings.
    # ★ 默认 False 是「安全默认值」：如果 provider 有 shell 权限，Agent 就能绕过路径映射，
    # 那就不该声称自己能做到技能隔离。宁可少一个功能，也不要给一个假的安全承诺。
    supports_agent_skill_isolation: bool = False

    # 抽象方法：子类必须实现，否则实例化时直接 TypeError。这就是「契约」的强制力。
    @abstractmethod
    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """Acquire a sandbox environment and return its ID.

        Returns:
            The ID of the acquired sandbox environment.
        """
        pass

    # ★ 为什么要有这一层异步包装：Docker / 远程 provisioning 都是阻塞 IO，
    # 直接在 async 函数里调用会让整个事件循环「冻住」（其它请求也无法推进）。
    # asyncio.to_thread 把阻塞函数丢到线程池执行，事件循环照常运转。
    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        """Acquire a sandbox without blocking the event loop.

        Most sandbox providers expose a synchronous lifecycle API because local
        Docker/provisioner operations are blocking. Async runtimes should call
        this method so those blocking operations run in a worker thread instead
        of stalling the event loop.
        """
        return await asyncio.to_thread(self.acquire, thread_id, user_id=user_id)

    # 技能同步的默认实现是「空操作」：挂载型 provider 直接就能看到宿主机的路径，无需上传。
    # 上传型 provider 需要覆盖它来自行拷贝文件。
    def sync_agent_skills(
        self,
        sandbox_id: str,
        *,
        thread_id: str,
        user_id: str,
        projection: "SkillProjectionPaths",
    ) -> None:
        """Synchronize a prepared thread skill projection into a sandbox.

        Bind-mount providers observe the stable projection roots directly and
        use this no-op implementation. Upload-based providers override it.
        """

    # 异步版必须走 lease 层（run_sync_lifecycle_operation），而不是裸的 to_thread：
    # 这样在同步操作完成之前，没人能把这个沙箱回收掉，避免「同步到一半沙箱没了」。
    async def sync_agent_skills_async(
        self,
        sandbox_id: str,
        *,
        thread_id: str,
        user_id: str,
        projection: "SkillProjectionPaths",
    ) -> None:
        """Async wrapper that keeps lifecycle ownership until sync finishes."""
        await run_sync_lifecycle_operation(
            self.sync_agent_skills,
            sandbox_id,
            thread_id=thread_id,
            user_id=user_id,
            projection=projection,
        )

    # 抽象方法：按 id 取回已存在的沙箱；取不到返回 None（而不是抛异常），
    # 因为「沙箱过期」是正常业务状态，调用方会据此重新 acquire。
    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox environment by ID.

        Args:
            sandbox_id: The ID of the sandbox environment to retain.
        """
        pass

    # ★ 「归属校验」钩子：只有当沙箱确实属于这个 thread/user 时才返回它，否则返回 None。
    # 默认返回 None 是「fail closed」（失败即拒绝）——没实现就不给，避免越权访问别人的沙箱。
    # 用 del 参数是为了明确表示「基类故意不用这些参数」，同时避免 linter 报未使用变量。
    def get_scoped(
        self,
        sandbox_id: str,
        *,
        thread_id: str,
        user_id: str,
    ) -> Sandbox | None:
        """Return an active sandbox only when it belongs to this identity.

        This hook must remain a non-blocking in-memory lookup. Providers that
        do not implement identity-aware lookup fail closed; the caller then
        resolves the canonical sandbox through ``acquire``.
        """
        del sandbox_id, thread_id, user_id
        return None

    # 抽象方法：释放 / 销毁沙箱。与 acquire 成对出现，构成完整生命周期。
    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """Release a sandbox environment.

        Args:
            sandbox_id: The ID of the sandbox environment to destroy.
        """
        pass

    # 重置：清掉「跨实例存活」的模块级缓存。改了配置要生效，就必须走这里，
    # 否则下一次 acquire 仍会拿到旧配置的沙箱。
    def reset(self) -> None:
        """Clear cached state that survives provider instance replacement.

        Provider overrides can release resources and make the instance unusable.
        """
        pass

    # 出网策略相关的一组钩子。默认值代表「不托管网络策略」的 provider：
    # 网络直连(open)、授权临时放行 300 秒、没有待审批事件。子类按需覆盖。
    def sandbox_network_mode(self) -> str:
        """Return the provider's effective outbound network mode."""
        return "open"

    def sandbox_network_temporary_grant_ttl(self) -> int:
        return 300

    def consume_network_policy_events(self, sandbox_id: str) -> list[dict[str, object]]:
        """Claim the oldest unsurfaced trusted-proxy event for a sandbox.

        Providers without a managed network policy use the empty default.
        """
        del sandbox_id
        return []

    # async 变体全部复用同一个「租约保护」包装，保证同步回调执行期间沙箱不会被并发回收。
    async def consume_network_policy_events_async(self, sandbox_id: str) -> list[dict[str, object]]:
        return await run_sync_lifecycle_operation(
            self.consume_network_policy_events,
            sandbox_id,
        )

    def deny_pending_network_policy_events(self, sandbox_id: str) -> bool:
        """Atomically deny all unsurfaced trusted-proxy events for a sandbox."""
        del sandbox_id
        return False

    async def deny_pending_network_policy_events_async(self, sandbox_id: str) -> bool:
        return await run_sync_lifecycle_operation(
            self.deny_pending_network_policy_events,
            sandbox_id,
        )

    def decide_network_policy_request(self, sandbox_id: str, request_id: str, decision: str) -> bool:
        """Apply a user decision to one trusted-proxy event."""
        del sandbox_id, request_id, decision
        return False

    async def decide_network_policy_request_async(self, sandbox_id: str, request_id: str, decision: str) -> bool:
        return await run_sync_lifecycle_operation(
            self.decide_network_policy_request,
            sandbox_id,
            request_id,
            decision,
        )


# ● 模块级全局变量 = 单例的「槽位」。None 表示尚未创建。
_default_sandbox_provider: SandboxProvider | None = None
# Guards every read and write of `_default_sandbox_provider`. The singleton is
# reachable from more than one OS thread (e.g. the main event loop and the Feishu
# channel thread, which runs its own loop), so a bare check-then-create can double
# initialize the provider, and an unsynchronized reset/shutdown racing a get can
# hand a caller `None` or a torn instance. Every access to the global below takes
# this lock, including the read+return in `get_sandbox_provider()`.
#
# The lock guards only the reference swap. Provider callbacks (`__init__`,
# `reset()`, `shutdown()`) and the dynamic import in `resolve_class()` run
# *outside* the lock: they are plugin-supplied (`config.sandbox.use` resolves to
# an arbitrary class) and may be slow or, worse, re-enter these lifecycle
# functions. Holding a non-reentrant `threading.Lock` across them would
# self-deadlock such a provider and would block every concurrent `get()` during a
# slow teardown. Keeping callbacks off the lock avoids both.
# ★ 这把锁只保护「把变量换成另一个对象」这一瞬间，不包住构造/销毁。
# 如果包住了：provider 的 __init__ 或 reset() 里又调用 get_sandbox_provider()，
# 就会「自己等自己」造成死锁（threading.Lock 不可重入）。这是个很典型的取舍。
_provider_lock = threading.Lock()

# Number of hosts (the Gateway lifespan) currently owning provider shutdown.
# Providers built while it is positive leave SIGTERM/SIGINT to the host, which
# drains in-flight runs before calling `shutdown_sandbox_provider()`. Guarded by
# `_provider_lock`; read by provider constructors, never under it.
# ● 计数器：记录「有多少个宿主(比如 Gateway 的 lifespan) 声称为关闭负责」。
# 大于 0 时，新建的 provider 就不去注册 SIGTERM/SIGINT 信号处理，
# 由宿主在「排空进行中的任务」之后再统一关闭，避免沙箱在任务还没跑完时被提前销毁。
_host_shutdown_owners = 0


# @contextmanager 把一个「有进入/退出两阶段」的函数包装成 with 语句可用的对象。
@contextmanager
def hold_sandbox_shutdown_ownership() -> Iterator[None]:
    """Declare that the host, not provider signal handlers, shuts sandboxes down.

    Enter this before any provider can be built. Providers built inside it skip
    their SIGTERM/SIGINT handlers and keep the SIGHUP fallback, since uvicorn
    does not handle SIGHUP. A provider built earlier keeps the handlers it
    already installed.
    """
    global _host_shutdown_owners
    # 进入 with 时 +1，退出时 -1（放在 finally 里，保证异常路径也会归还）。
    with _provider_lock:
        _host_shutdown_owners += 1
    try:
        yield
    finally:
        with _provider_lock:
            _host_shutdown_owners -= 1


# 只读查询：不改全局状态，返回是否已有宿主接管关闭职责。
def host_owns_sandbox_shutdown() -> bool:
    """Return whether a host currently owns sandbox provider shutdown."""
    with _provider_lock:
        return _host_shutdown_owners > 0


# 「只读、不创建」的取用方式：给那些只想看看有没有初始化过的调用方用，
# 避免它们顺手把 provider 建起来（创建可能有副作用）。
def get_initialized_sandbox_provider() -> SandboxProvider | None:
    """Return the provider only when another lifecycle path initialized it."""
    with _provider_lock:
        return _default_sandbox_provider


# ● 单例入口（工厂 + 缓存）。第一次调用会：读配置 → 动态 import 出实现类 → 实例化 → 缓存。
# 之后所有调用直接返回缓存，所以这个方法会被调用得非常频繁，必须尽量轻。
def get_sandbox_provider(**kwargs) -> SandboxProvider:
    """Get the sandbox provider singleton.

    Returns a cached singleton instance. Use `reset_sandbox_provider()` to clear
    the cache, or `shutdown_sandbox_provider()` to properly shutdown and clear.

    Returns:
        A sandbox provider instance.
    """
    global _default_sandbox_provider
    # Fast path: a single locked read so a concurrent reset/shutdown can't null
    # the global between the check and the return.
    with _provider_lock:
        if _default_sandbox_provider is not None:
            return _default_sandbox_provider

    # Cold start. Resolve + construct outside the lock: the import and the
    # provider constructor are plugin code and must not run under a non-reentrant
    # lock. The construction may race another caller; we reconcile under the lock.
    config = get_app_config()
    # ★ 动态 import 的核心一行：config.sandbox.use 长得像 "deerflow.sandbox.local:LocalSandboxProvider"，
    # resolve_class 会在运行期按字符串找到类，并校验它确实是 SandboxProvider 的子类。
    cls = resolve_class(config.sandbox.use, SandboxProvider)
    provider = cls(**kwargs)

    # 第二次加锁：防止两个线程同时冷启动，把两个实例都装上（这就是「双重检查」）。
    with _provider_lock:
        if _default_sandbox_provider is None:
            _default_sandbox_provider = provider
            return provider
        # We lost the install race: another thread got there first. `winner` is
        # read under the same lock, so it is always a live instance, never None.
        winner = _default_sandbox_provider

    # Discard the instance we just built (outside the lock). For providers with
    # side-effectful constructors (e.g. AioSandboxProvider starts an idle-checker
    # thread), this tears down the orphan so it does not leak — issue #3721.
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    return winner


# 只清缓存、不销毁：下一次 get 会重新建一个。多用于测试或切换配置。
def reset_sandbox_provider() -> None:
    """Reset the sandbox provider singleton.

    This clears the cached instance without calling shutdown directly.
    The next call to `get_sandbox_provider()` will create a new instance.
    Useful for testing or when switching configurations.

    Providers can override `reset()` to clear any module-level state they keep
    alive across instances (for example, `LocalSandboxProvider`'s cached
    `LocalSandbox` singleton). Without it, config/mount changes would not take
    effect on the next acquire().

    A provider override can release active sandboxes during reset.
    Otherwise, active sandboxes become orphaned.
    Do not reuse the detached provider after reset.
    Use `shutdown_sandbox_provider()` for proper cleanup.
    """
    global _default_sandbox_provider
    # Detach the reference under the lock, then run the provider's `reset()`
    # callback outside it (see the `_provider_lock` note).
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is not None:
        from deerflow.sandbox.lease import discard_sandbox_lease_manager

        discard_sandbox_lease_manager(provider)
        provider.reset()


# 规范关闭：清缓存 + 调 shutdown() + 回收租约。生产环境退出时应该用这个，而不是 reset。
def shutdown_sandbox_provider() -> None:
    """Shutdown the provider, keeping failed cleanup reachable for retry.

    A provider installed concurrently takes precedence over a failed instance.
    Provider callbacks run outside the registry lock; discard execution leases
    only after successful shutdown.
    """
    global _default_sandbox_provider
    # Detach the reference under the lock, then run the (potentially slow)
    # `shutdown()` callback outside it (see the `_provider_lock` note).
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is not None and hasattr(provider, "shutdown"):
        from deerflow.sandbox.lease import discard_sandbox_lease_manager

        try:
            provider.shutdown()
        except BaseException:
            # 用 BaseException 而不是 Exception：连 Ctrl+C(CancelledError/KeyboardInterrupt) 也要
            # 把实例放回槽位，否则这次清理失败后 provider 就彻底找不回来了，无法重试。
            # Keep a failed cleanup reachable for a later shutdown call, without
            # replacing a new provider installed while this callback ran.
            with _provider_lock:
                if _default_sandbox_provider is None:
                    _default_sandbox_provider = provider
            raise
        else:
            discard_sandbox_lease_manager(provider)


# 测试 / 依赖注入入口：直接塞一个自定义或 mock 实例进来。
# 注意旧实例只是被替换，不会被关闭——生命周期由调用方自己负责。
def set_sandbox_provider(provider: SandboxProvider) -> None:
    """Set a custom sandbox provider instance.

    This allows injecting a custom or mock provider for testing purposes.

    Note: any previously installed provider is replaced but not shut down; the
    caller owns the lifecycle of the instance it is overwriting.

    Args:
        provider: The SandboxProvider instance to use.
    """
    global _default_sandbox_provider
    with _provider_lock:
        previous = _default_sandbox_provider
        _default_sandbox_provider = provider
    if previous is not None and previous is not provider:
        from deerflow.sandbox.lease import discard_sandbox_lease_manager

        discard_sandbox_lease_manager(previous)
