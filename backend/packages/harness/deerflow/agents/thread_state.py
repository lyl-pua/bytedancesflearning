# =============================================================================
# 【中文导读 · 初学者版】
#
# ★ 这个文件在整条链路的位置
#   它是整张 Agent 图「唯一的那块共享黑板」的定义处：
#   图里每个节点/中间件返回的 dict，都会被按本文件声明的规则合并进 ThreadState。
#   checkpointer 存档存的、thread_id 隔离的，也正是这份状态。
#
# ★ 它解决什么问题
#   多个节点并发地往同一个字段写值时，「谁覆盖谁」必须事先讲清楚，
#   否则结果随机。这个文件用「reducer（合并函数）」把每个字段的合并语义
#   固定下来：追加、合并去重、只保留最后一次非空……
#   没有 reducer 的字段则采用「后写覆盖」（last-write-wins）。
#
# ● 涉及的知识点（编号说明）
#   1. TypedDict：Python 里给 dict 加「字段名 + 类型」标注的方式，
#      LangGraph 用它来描述状态结构（有哪些键、各是什么类型）。
#   2. Annotated[类型, reducer]：Annotated 只是给类型挂「附加元信息」，
#      LangGraph 约定这里的元信息是一个 reducer 函数。写法：
#        字段: Annotated[list[str], merge_artifacts]
#      意思是：当某节点返回 {"字段": 新值}，不要覆盖，而是调用
#      merge_artifacts(旧值, 新值) 得到合并结果。
#   3. 消息通道与 add_messages：messages 字段天然要「追加」，
#      并且支持用 RemoveMessage 按 id 删除某条消息（上下文压缩就靠它）。
#      因此消息字段的 reducer 必须复刻这些语义。
#   4. NotRequired：表示该键可以不存在（可选字段），读取时要按 None 处理。
#   5. 检查点（checkpoint）与 full / delta 两种模式：
#      full   = 每步存整份状态，简单但体积大；
#      delta  = 只存「这一步的增量」，靠 DeltaChannel 累积，省空间但读起来绕。
#      本文件两种都提供，并在 get_thread_state_schema() 里按模式选一个。
#   6. 幂等（idempotent）：同一写操作执行两次结果相同。很多 reducer 都
#      刻意做成幂等（如 sandbox 只接受相同 id 的重复写），这样重放安全。
# =============================================================================

import copy
import uuid
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Annotated, Any, NotRequired, TypedDict, cast, get_type_hints

from langchain.agents import AgentState
from langchain_core.messages import (
    AnyMessage,
    BaseMessageChunk,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langgraph.channels import DeltaChannel
from langgraph.graph.message import REMOVE_ALL_MESSAGES

import deerflow.checkpoint_patches as _checkpoint_patches  # noqa: F401 - import-time saver fixes
from deerflow.agents.goal_state import GoalOutcomeState, GoalState
from deerflow.agents.task_continuity.state import TaskNotesChannel, merge_task_notes
from deerflow.config.database_config import DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY, CheckpointChannelMode
from deerflow.subagents.status_contract import SUBAGENT_STATUS_VALUES


def _resolve_snapshot_frequency(snapshot_frequency: int | None) -> int:
    """Resolve the effective cadence: explicit value, else process-frozen,
    else default. Imported lazily — ``deerflow.runtime.__init__`` reaches this
    module via ``checkpoint_state``, so a top-level import would cycle."""
    if snapshot_frequency is not None:
        return snapshot_frequency
    from deerflow.runtime.checkpoint_mode import resolve_checkpoint_snapshot_frequency

    return resolve_checkpoint_snapshot_frequency()


class SandboxState(TypedDict):
    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]


class BackgroundTaskState(TypedDict):
    task_id: str
    task_name: str
    status: str
    updated_at: str


class BlobRefData(TypedDict):
    """JSON-safe checkpoint representation of ``deerflow.storage.BlobRef``."""

    sha256: str
    size: int
    kind: str
    content_type: NotRequired[str | None]


class ViewedImageData(TypedDict):
    """Metadata for a viewed image file.

    Only lightweight metadata is persisted in checkpoint state. Image bytes are
    read on-demand from the active sandbox or from a synchronized host copy whose
    size and SHA-256 match the previously viewed bytes. This avoids duplicating
    large base64 payloads across every checkpoint (see #4138).
    """

    mime_type: str
    size: int
    actual_path: str
    sha256: str
    source_sandbox_id: NotRequired[str]
    blob_ref: NotRequired[BlobRefData]


# ★ 从第一个 reducer 开始，请用「合并函数」的眼光读每个函数：
#   签名永远是 (旧值, 新值) -> 合并后的值，返回值会写回状态。
#   ● 三条通用约定（后面每个函数都遵守）：
#     - new 为 None  → 表示「这个节点根本没碰这个字段」，保留旧值。
#     - new 为空集合 → 具体语义各不相同（有的表示清空，有的表示无操作），
#       所以每个函数都会显式说明，不能一概而论。
#     - 冲突不可调和时 → 直接抛错（fail closed），绝不「猜一个」，
#       因为悄悄选一个会让状态错误延迟到很久以后才暴露。
def merge_sandbox(existing: SandboxState | None, new: SandboxState | None) -> SandboxState | None:
    """Reducer for sandbox state - accepts idempotent writes only.

    Multiple sandbox tools can initialize lazily in the same graph step and
    emit the same sandbox_id via Command(update=...). LangGraph needs an
    explicit reducer for that shared state key. Different sandbox ids in the
    same thread indicate a lifecycle/isolation bug, so fail closed instead of
    choosing one silently.
    """
    if new is None:
        return existing
    if existing is None:
        return new

    existing_id = existing.get("sandbox_id")
    new_id = new.get("sandbox_id")
    if existing_id == new_id:
        return existing
    raise ValueError(f"Conflicting sandbox state updates: {existing_id!r} != {new_id!r}")


SandboxStateField = Annotated[NotRequired[SandboxState | None], merge_sandbox]


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer for artifacts list - merges and deduplicates artifacts."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    # Use dict.fromkeys to deduplicate while preserving order
    return list(dict.fromkeys(existing + new))


def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """Reducer for viewed_images dict - merges image dictionaries.

    Special case: If new is an empty dict {}, it clears the existing images.
    This allows middlewares to clear the viewed_images state after processing.

    Dict order tracks recency: a key updated by ``new`` is re-inserted at the
    end, so a re-viewed image becomes the newest entry. The image injection
    middleware admits entries newest-first and tells the model to re-view an
    omitted image, so the recovery only works if the re-viewed entry actually
    moves to the newest position (#6825, review on #6829).
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # Special case: empty dict means clear all viewed images
    if len(new) == 0:
        return {}
    # Merge dictionaries; new values override existing ones for same keys, and
    # updated keys move to the end so insertion order reflects recency.
    merged = {key: value for key, value in existing.items() if key not in new}
    merged.update(new)
    return merged


def merge_todos(existing: list | None, new: list | None) -> list | None:
    """Reducer for todos list - keeps the last non-None value.

    Semantics:
    - If `new` is None (node didn't touch todos), preserve `existing`.
    - If `new` is provided (even empty list), it represents an explicit
      update and wins over `existing`.
    """
    if new is None:
        return existing
    return new


def merge_goal(existing: GoalState | None, new: GoalState | None) -> GoalState | None:
    """Reducer for goal state - preserves existing when a node does not touch it."""
    if new is None:
        return existing
    return new


class PromotedTools(TypedDict):
    catalog_hash: str
    names: list[str]
    turn_id: NotRequired[str]


def merge_promoted(existing: PromotedTools | None, new: PromotedTools | None) -> PromotedTools | None:
    """Reducer for deferred-tool promotions, scoped by catalog and user turn.

    - new None/empty -> preserve existing (node didn't touch promotions).
    - catalog_hash or explicit turn_id changed -> replace wholesale, dropping
      stale names.
    - same scope -> union names, dedupe, preserve order.
    - a missing new turn_id inherits the existing scope. ``tool_search`` writes
      this shape after the filter establishes the current turn before the model.
    """
    if not new:
        return existing
    new_turn_id = new.get("turn_id")
    replace = existing is None or existing.get("catalog_hash") != new["catalog_hash"] or (new_turn_id is not None and existing.get("turn_id") != new_turn_id)
    existing_names = existing.get("names") if existing is not None else None
    if not isinstance(existing_names, list) or not all(isinstance(name, str) for name in existing_names):
        existing_names = []
    names = new["names"] if replace else existing_names + new["names"]
    merged: PromotedTools = {
        "catalog_hash": new["catalog_hash"],
        "names": list(dict.fromkeys(names)),
    }
    turn_id = new_turn_id if new_turn_id is not None else (existing or {}).get("turn_id")
    if turn_id is not None:
        merged["turn_id"] = turn_id
    return merged


TERMINAL_STATUSES: frozenset[str] = frozenset(SUBAGENT_STATUS_VALUES)
_DELEGATION_LEDGER_MAX_ENTRIES = 50


class DelegationEntry(TypedDict):
    id: str
    run_id: NotRequired[str]
    description: str
    subagent_type: str
    status: str
    result_brief: NotRequired[str]
    result_sha256: NotRequired[str]
    result_ref: NotRequired[str]
    # Why a guardrail cap ended the run early (#3875 Phase 2): token_capped /
    # turn_capped / loop_capped. The status stays completed/failed; this field
    # is the additive signal that distinguishes a capped run from a clean one.
    stop_reason: NotRequired[str]
    # RFC #4651 PR2: parent-side citation-check verdict (advisory execution
    # evidence), stamped at task write-back; absent on legacy history.
    receipt_verdict: NotRequired[dict]
    # RFC #4651 PR4: deterministic acceptance-checklist verdict, same
    # provenance as receipt_verdict.
    acceptance_verdict: NotRequired[dict]
    created_at: str


def merge_delegations(existing: list[DelegationEntry] | None, new: list[DelegationEntry] | None) -> list[DelegationEntry]:
    """Reducer for the delegation ledger.

    - new None/empty -> preserve existing.
    - append entries, replacing the same (run_id, tool-call id) with the latest
      version while preserving first-seen order. Provider ids can repeat in
      later runs of the same thread.
    - terminal status is never overwritten by a non-terminal status.
    """
    if not new:
        return existing or []

    by_key: dict[tuple[str | None, str], DelegationEntry] = {}
    order: list[tuple[str | None, str]] = []
    for entry in [*(existing or []), *new]:
        entry_id = entry["id"]
        key = (entry.get("run_id") or None, entry_id)
        if key[0] is None:
            # Legacy updates without run_id still update the most recent
            # matching entry, as they did before run-scoped identities.
            key = next((prior for prior in reversed(order) if prior[1] == entry_id), key)
        previous = by_key.get(key)
        if previous is not None and previous["status"] in TERMINAL_STATUSES and entry["status"] not in TERMINAL_STATUSES:
            continue
        if key not in by_key:
            order.append(key)
        elif previous.get("created_at"):
            entry = {**entry, "created_at": previous["created_at"]}
            if previous.get("run_id") and not entry.get("run_id"):
                entry["run_id"] = previous["run_id"]
        by_key[key] = entry
    merged = [by_key[key] for key in order]
    if len(merged) > _DELEGATION_LEDGER_MAX_ENTRIES:
        merged = merged[-_DELEGATION_LEDGER_MAX_ENTRIES:]
    return merged


_SKILL_CONTEXT_MAX_ENTRIES = 8
_SKILL_DESCRIPTION_MAX_CHARS = 500


class SkillEntry(TypedDict):
    name: str
    path: str
    description: str
    loaded_at: int


def _normalize_skill_entry(entry: Mapping[str, object]) -> SkillEntry:
    """Drop legacy payload keys before storing skill_context back to state."""
    description = entry.get("description")
    loaded_at = entry.get("loaded_at")
    return {
        "name": str(entry.get("name") or ""),
        "path": str(entry["path"]),
        "description": " ".join(description.split())[:_SKILL_DESCRIPTION_MAX_CHARS] if isinstance(description, str) else "",
        "loaded_at": loaded_at if isinstance(loaded_at, int) else 0,
    }


def merge_skill_context(existing: list[SkillEntry] | None, new: list[SkillEntry] | None) -> list[SkillEntry]:
    """Reducer for the skill-context channel.

    - new None/empty -> preserve existing.
    - legacy entries are normalized to references; verbatim body keys are dropped.
    - dedup by ``path``; later reads refresh recency and replace the reference.
    - cap by keeping the most recently read entries. ``loaded_at`` is
      observational only because message indices reset after compaction.
    """
    normalized_existing = [_normalize_skill_entry(entry) for entry in existing or []]
    if not new:
        return normalized_existing

    by_path: dict[str, SkillEntry] = {}
    order: list[str] = []
    for entry in normalized_existing:
        path = entry["path"]
        if path not in by_path:
            order.append(path)
        by_path[path] = entry

    for entry in (_normalize_skill_entry(entry) for entry in new):
        path = entry["path"]
        if path in by_path:
            order.remove(path)
        order.append(path)
        by_path[path] = entry

    merged = [by_path[path] for path in order]
    if len(merged) > _SKILL_CONTEXT_MAX_ENTRIES:
        merged = merged[-_SKILL_CONTEXT_MAX_ENTRIES:]
    return merged


# Absolute ceiling applied by the reducer itself. The operator-configured cap
# (`tool_artifacts.max_entries`) is enforced per-agent by
# ArtifactCaptureMiddleware when it emits updates, so two agents in one process
# can carry different caps without sharing state.
_ARTIFACT_MAX_ENTRIES_CEILING = 1000


class ArtifactEntry(TypedDict):
    """A captured artifact reference from a tool result (issue #4676).

    Stored in ``ThreadState.tool_artifacts`` so it survives context compaction.
    The model sees only the short ``handle``; the real reference (path, URL,
    task id) lives here and is resolved at tool-call time.
    """

    handle: str
    tool_name: str
    tool_call_id: str
    call_index: int
    artifact_type: str
    display_name: str
    real_ref: str
    mime_type: NotRequired[str]
    created_at: str
    consumed_by: NotRequired[list[str]]


def merge_tool_artifacts(existing: list[ArtifactEntry] | None, new: list[ArtifactEntry] | None) -> list[ArtifactEntry]:
    """Reducer for the tool-artifact registry channel.

    - new None/empty -> preserve existing.
    - append entries, replacing same handle with the latest version while
      preserving first-seen order (latest wins, e.g. for consumption updates).
    - a trailing ``{"op": "trim_to", "keep": N}`` directive (emitted by the
      capture middleware) makes the configured cap a sliding window: the
      oldest entries beyond N are evicted. Without a directive nothing is
      evicted, so updates stay purely additive.
    - absolute ceiling of 1000 applies regardless.
    """
    if not new:
        return existing or []

    by_handle: dict[str, ArtifactEntry] = {}
    order: list[str] = []
    for entry in [*(existing or []), *new]:
        if not isinstance(entry, dict) or "handle" not in entry:
            continue
        handle = entry["handle"]
        if handle not in by_handle:
            order.append(handle)
        by_handle[handle] = entry

    merged = [by_handle[handle] for handle in order]

    for item in reversed(new):
        if isinstance(item, dict) and item.get("op") == "trim_to":
            keep = min(int(item.get("keep", len(merged))), _ARTIFACT_MAX_ENTRIES_CEILING)
            if keep < len(merged):
                merged = merged[-keep:]
            break

    if len(merged) > _ARTIFACT_MAX_ENTRIES_CEILING:
        merged = merged[-_ARTIFACT_MAX_ENTRIES_CEILING:]
    return merged


# ★ 整个状态结构的总定义（所有字段 + 各自的合并规则一览）
#   ● 标注格式：字段名: Annotated[类型, reducer]
#     - 有 Annotated 的字段 → 走自定义合并逻辑（追加 / 去重 / 保留最后一次）；
#     - 只有普通类型（如 title: NotRequired[str | None]）→ 后写覆盖。
#   ● 为什么区分「可覆盖」与「必须合并」：
#     - title、summary_text、goal_outcome 这类「单值事实」，新值覆盖旧值是对的；
#     - messages、artifacts、delegations 这类「集合」，覆盖会丢数据，
#       必须用 reducer 累积。
#   ● 几个要点字段（初学者最容易困惑的）：
#     messages        对话历史（继承自 AgentState，用 add_messages 语义）
#     summary_text    上下文压缩后留下的「摘要」，原文被 RemoveMessage 删掉
#     delegations     子智能体委派台账（谁派了谁、状态如何）
#     skill_context   已加载技能的索引（渐进式披露的「已展开记录」）
#     tool_artifacts  工具产物登记表（大产物只留 handle，真实引用存这里）
#     task_notes      任务连续性笔记
class ThreadState(AgentState):
    sandbox: SandboxStateField
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list[str], merge_artifacts]
    todos: Annotated[list | None, merge_todos]
    goal: Annotated[GoalState | None, merge_goal]
    goal_outcome: NotRequired[GoalOutcomeState | None]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # image_path -> metadata (no base64)
    promoted: Annotated[PromotedTools | None, merge_promoted]
    delegations: Annotated[list[DelegationEntry], merge_delegations]
    skill_context: Annotated[list[SkillEntry], merge_skill_context]
    tool_artifacts: Annotated[list[ArtifactEntry], merge_tool_artifacts]
    tool_artifact_processed: Annotated[list[str], merge_artifacts]
    task_notes: Annotated[dict | None, TaskNotesChannel(dict | None, merge_task_notes)]
    task_history: NotRequired[dict | None]
    summary_text: NotRequired[str | None]
    background_tasks: NotRequired[list[BackgroundTaskState]]


def _normalize_messages(value: Any) -> list[AnyMessage]:
    values = value if isinstance(value, list) else [value]
    messages = [message_chunk_to_message(cast(BaseMessageChunk, message)) for message in convert_to_messages(values)]
    for message in messages:
        if message.id is None:
            message.id = str(uuid.uuid4())
    return messages


def _index_messages(
    messages: list[AnyMessage | None],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    latest_position: dict[str, int] = {}
    positions_by_id: dict[str, list[int]] = {}
    for position, message in enumerate(messages):
        if message is None:
            continue
        message_id = cast(str, message.id)
        latest_position[message_id] = position
        positions_by_id.setdefault(message_id, []).append(position)
    return latest_position, positions_by_id


def _raise_null_write(has_messages: bool) -> None:
    # ``add_messages(left, None)`` reports only ``left`` when the accumulated
    # message list is non-empty; with an empty list, it reports only ``right``.
    received = "left" if has_messages else "right"
    raise ValueError(f"Must specify non-null arguments for both 'left' and 'right'. Only received: '{received}'.")


# ★ delta 模式下的消息通道（本文件里最「进阶」的一段）
#   背景：一份对话历史动辄几万 token，若每步都整份存进检查点，
#   存储与网络开销都会爆炸。DeltaChannel 的做法是：
#     只记录「这一步新增/删除了哪些消息」，读的时候再把增量依次叠加还原。
#   ● merge_message_writes 就是在做「叠加」：它把一批增量写（writes）
#     按 add_messages 的语义折进已有列表，要点：
#     - 新消息 append；同 id 的消息替换（相当于「编辑」）；
#     - RemoveMessage(id=...) 删除指定消息；
#     - RemoveMessage(id=REMOVE_ALL_MESSAGES) 清空全部（压缩时用），
#       它之后的写才生效；
#     - 删除不存在的 id 直接报错（防止历史分叉导致状态错乱）。
#   ● 为什么要自己实现而不用 LangGraph 私有实现：私有实现不保证
#     完整的转换、id 补齐与删除语义，一旦升级就可能行为不一致。
def merge_message_writes(state: list[AnyMessage], writes: Sequence[Any]) -> list[AnyMessage]:
    """Fold DeltaChannel writes with ``add_messages`` semantics in linear time.

    LangGraph's private ``_messages_delta_reducer`` is also linear, but does
    not preserve the public reducer's full coercion, ID, removal, and
    ``REMOVE_ALL_MESSAGES`` behavior.
    """
    if not writes:
        return list(state)
    if writes[0] is None:
        _raise_null_write(bool(state))

    messages: list[AnyMessage | None] = _normalize_messages(state)
    latest_position, positions_by_id = _index_messages(messages)

    for write in writes:
        if write is None:
            _raise_null_write(bool(latest_position))
        normalized_write = _normalize_messages(write)
        remove_all_idx = None
        for position, message in enumerate(normalized_write):
            if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES:
                remove_all_idx = position

        if remove_all_idx is not None:
            messages = list(normalized_write[remove_all_idx + 1 :])
            latest_position, positions_by_id = _index_messages(messages)
            continue

        ids_to_remove: set[str] = set()
        for message in normalized_write:
            message_id = cast(str, message.id)
            existing_position = latest_position.get(message_id)
            if existing_position is not None:
                if isinstance(message, RemoveMessage):
                    ids_to_remove.add(message_id)
                else:
                    ids_to_remove.discard(message_id)
                    messages[existing_position] = message
                continue

            if isinstance(message, RemoveMessage):
                raise ValueError(f"Attempting to delete a message with an ID that doesn't exist ('{message_id}')")

            position = len(messages)
            messages.append(message)
            latest_position[message_id] = position
            positions_by_id[message_id] = [position]

        for message_id in ids_to_remove:
            for position in positions_by_id.pop(message_id):
                messages[position] = None
            del latest_position[message_id]

    return [message for message in messages if message is not None]


def delta_messages_field(snapshot_frequency: int = DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY) -> Any:
    """Messages field annotation with a ``DeltaChannel`` at the given cadence."""
    return Annotated[
        list[AnyMessage],
        DeltaChannel(merge_message_writes, snapshot_frequency=snapshot_frequency),
    ]


DELTA_MESSAGES_FIELD = delta_messages_field()


class DeltaThreadState(ThreadState):
    messages: DELTA_MESSAGES_FIELD


THREAD_STATE_REDUCER_FIELDS = frozenset(
    {
        "messages",
        "sandbox",
        "artifacts",
        "todos",
        "goal",
        "viewed_images",
        "promoted",
        "delegations",
        "skill_context",
        "tool_artifacts",
        "tool_artifact_processed",
        "task_notes",
    }
)


# ★ 按模式选状态结构：full → ThreadState；delta → DeltaThreadState
#   （DeltaThreadState = ThreadState 的基础上把 messages 换成 DeltaChannel）
#   ● 为什么要按快照频率做缓存（@cache）：不同频率要生成不同的 TypedDict 类，
#     而类对象应该被复用（同一个频率永远拿到同一个类），否则类型判断会失效。
#   ● 下面三个函数分别服务于三种调用方：
#     get_thread_state_schema       → 默认状态结构
#     adapt_state_schema_for_mode   → 调用方自定义 schema 时做适配
#     normalize_middleware_state_schemas → 中间件自带的 schema 也做适配
def get_thread_state_schema(mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> type:
    if mode != "delta":
        return ThreadState
    return _delta_thread_state_schema(_resolve_snapshot_frequency(snapshot_frequency))


@cache
def _delta_thread_state_schema(snapshot_frequency: int) -> type:
    """Delta thread schema keyed by cadence; the default keeps the static
    ``DeltaThreadState`` identity so existing type checks keep holding."""
    if snapshot_frequency == DEFAULT_CHECKPOINT_SNAPSHOT_FREQUENCY:
        return DeltaThreadState
    annotations = get_type_hints(ThreadState, include_extras=True)
    annotations["messages"] = delta_messages_field(snapshot_frequency)
    return TypedDict(
        f"DeltaThreadState_f{snapshot_frequency}",
        annotations,
        total=getattr(ThreadState, "__total__", True),
    )


def adapt_state_schema_for_mode(schema: type, mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> type:
    if mode == "full":
        return schema
    return _adapt_state_schema_for_delta(schema, _resolve_snapshot_frequency(snapshot_frequency))


@cache
def _adapt_state_schema_for_delta(schema: type, snapshot_frequency: int) -> type:
    annotations = get_type_hints(schema, include_extras=True)
    annotations["messages"] = delta_messages_field(snapshot_frequency)
    return TypedDict(
        f"Delta{schema.__module__.replace('.', '_')}_{schema.__name__}_f{snapshot_frequency}",
        annotations,
        total=getattr(schema, "__total__", True),
    )


def normalize_middleware_state_schemas(middleware: Sequence[Any], mode: CheckpointChannelMode, snapshot_frequency: int | None = None) -> list[Any]:
    if mode == "full":
        return list(middleware)
    resolved_frequency = _resolve_snapshot_frequency(snapshot_frequency)
    normalized = []
    for item in middleware:
        schema = getattr(item, "state_schema", None)
        if schema is None:
            normalized.append(item)
            continue
        adapted = copy.copy(item)
        adapted.state_schema = adapt_state_schema_for_mode(schema, mode, resolved_frequency)
        normalized.append(adapted)
    return normalized
