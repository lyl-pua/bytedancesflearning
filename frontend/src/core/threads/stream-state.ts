/**
 * 【中文导读 · 初学者版】
 *
 * ● 这个文件做什么：
 *   后端智能体（LangGraph）流式运行时，会不断推送「状态更新帧(updates)」。
 *   每一帧只说明「某个节点改动了哪些字段」，并不是完整快照。
 *   本文件把这些零散的增量「合并/折叠」成前端可以直接渲染的线程状态。
 *
 * ● 在链路里的位置：
 *   SSE 流式事件 → 解析出 updates 帧 → reduceThreadStateUpdates()
 *   → 合并进 thread.values（标题 / 产物 / 待办 / 目标）→ 消息列表、目标条、待办列表渲染。
 *
 * ● 涉及的知识点：
 *   1. 流式增量渲染(SSE)：数据一段段到达，界面逐步更新，而不是等全部跑完再刷一次。
 *   2. 事件合并(reducer 语义)：后端用什么规则合并（追加去重、null 表示「本次不改」），
 *      前端就必须照抄同样的规则，否则界面显示的和服务端真实状态会打架。
 *   3. 不可变数据：返回新数组/新对象，不原地修改旧值，React 才能察觉变化并重渲染。
 *   4. 类型守卫(type guard)：isRecord / isGoalState 把 unknown 收窄成具体类型，安全访问字段。
 *   5. 为什么排除 messages：消息的增量拼接由 SDK 的 messages-tuple 管理器负责，
 *      这里再合并一次会造成消息重复。
 */

import type { AgentThreadState, ArtifactEntry, GoalState } from "./types";

// 中文：一次更新只包含「部分字段」，所以用 Partial 描述这种「补丁」。
type ThreadStatePatch = Partial<AgentThreadState>;

// 类型守卫：把 unknown 收窄成「普通对象」。数组和 null 都算「不是记录」，
// 因为后续要做 Object.hasOwn 之类的字段检查，对数组做这些检查没有意义。
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * An active goal as the backend judges one (`status` "active"), so a
 * malformed or inactive `goal` in history, worker frames or a status read
 * never renders, and the bar and the edit lock agree with the server. POST
 * /state and run input can store one without `created_at`/`updated_at`.
 * `goal_outcome` is not folded here: only the worker's clear writes it,
 * never through `updates`.
 */
// 中文：只有 status === "active" 且 objective 是字符串，才算「进行中的目标」。
// 后端会以多种形式回传 goal，前端统一在这里收窄，避免把脏数据渲染成进度条。
export function isGoalState(value: unknown): value is GoalState {
  return (
    isRecord(value) &&
    typeof value.objective === "string" &&
    value.status === "active"
  );
}

// 中文：这 5 个字段是「界面真的会渲染出来」的状态。
// 只有它们变了才值得提醒 UI 更新，其他内部字段变了可以忽略，省一次重渲染。
const RENDERED_THREAD_STATE_KEYS = [
  "title",
  "artifacts",
  "tool_artifacts",
  "todos",
  "goal",
];

// 中文：判断这一帧里「有没有值得渲染的字段变化」。
// data 是「按节点名分组」的对象，所以先用 Object.values 拆开，再逐个检查。
// Object.hasOwn（而不是 `in`）只认对象自身的键，避免命中原型链上的属性。
export function hasRenderedThreadStateUpdate(data: unknown): boolean {
  if (!isRecord(data)) return false;
  return Object.values(data).some(
    (update) =>
      isRecord(update) &&
      RENDERED_THREAD_STATE_KEYS.some((key) => Object.hasOwn(update, key)),
  );
}

// 中文：合并产物路径列表。返回 undefined 表示「这一帧的数据不可用，保持原状」；
// 返回新数组表示「合并成功」。这种「undefined = 不改」的约定贯穿整个文件。
function mergeArtifacts(
  existing: AgentThreadState["artifacts"],
  incoming: unknown,
): string[] | undefined {
  if (incoming == null || !Array.isArray(incoming)) {
    return undefined;
  }
  if (!incoming.every((path) => typeof path === "string")) {
    return undefined;
  }

  // 用 Set 去重（后端 reducer 是「追加 + 去重」，前端保持一致），
  // 并生成新数组（不可变更新），React 才会认为数据变了。
  return [...new Set([...(existing ?? []), ...incoming])];
}

// 中文：校验一条「工具产物」记录是否完整（6 个字符串字段 + 1 个整数字段）。
// 只要有一个字段类型不对就丢弃，宁可不显示也不显示半截数据。
function isArtifactEntry(value: unknown): value is ArtifactEntry {
  return (
    isRecord(value) &&
    [
      "handle",
      "tool_name",
      "tool_call_id",
      "artifact_type",
      "display_name",
      "real_ref",
    ].every((key) => typeof value[key] === "string") &&
    Number.isInteger(value.call_index)
  );
}

// 中文：合并「工具产物」列表。用 Map 按 handle 去重：
// 同一个 handle 再次出现时「原位替换元信息」，但不改变它首次出现的顺序。
function mergeToolArtifacts(
  existing: ArtifactEntry[] | undefined,
  incoming: unknown,
): ArtifactEntry[] | undefined {
  if (!Array.isArray(incoming)) return undefined;
  // Same-handle updates replace metadata without moving first-seen order.
  // Capture can append a retention directive after its entries.
  // Map 的 key 是 handle：先放的（旧）在前，后放的（新）会覆盖同名项但保留原顺序。
  const byHandle = new Map<string, ArtifactEntry>();
  for (const item of [...(existing ?? []), ...incoming]) {
    if (isArtifactEntry(item)) byHandle.set(item.handle, item);
  }
  // 中文：从后往前找「裁剪指令」trim_to（保留 keep 条），只认最后一条；
  // 默认上限 1000，防止产物列表无限增长把前端撑爆。
  let keep = 1000;
  for (let index = incoming.length - 1; index >= 0; index--) {
    const item: unknown = incoming[index];
    if (isRecord(item) && item.op === "trim_to") {
      if (
        typeof item.keep === "number" &&
        Number.isInteger(item.keep) &&
        item.keep > 0
      ) {
        keep = Math.min(item.keep, keep);
      }
      break;
    }
  }
  return [...byHandle.values()].slice(-keep);
}

/**
 * Fold a LangGraph `updates` frame into the state fields rendered by the chat
 * UI. Updates are grouped by node name and carry reducer inputs, not complete
 * state snapshots, so these fields must mirror the reducers in
 * `deerflow.agents.thread_state` rather than being shallowly assigned.
 *
 * `messages` is deliberately excluded. The SDK's `messages-tuple` manager owns
 * chunk assembly and same-id replacement; applying the node's messages update
 * through `mutate` as well would duplicate messages and bypass chunk merging.
 */
/**
 * 中文导读：把一个 updates 帧「折叠」成 UI 需要的补丁。
 * 返回 undefined 表示「这一帧没有任何可用变化，别动状态」。
 * 注意：这里只处理 title/artifacts/tool_artifacts/todos/goal 五类字段，
 * messages 故意不处理（见上方英文注释：交给 SDK 的消息管理器，避免重复）。
 */
export function reduceThreadStateUpdates(
  previous: AgentThreadState,
  data: unknown,
): ThreadStatePatch | undefined {
  if (!isRecord(data)) {
    return undefined;
  }

  // patch 是「增量补丁」，artifacts/toolArtifacts 额外用局部变量累计，
  // 因为同一帧可能有多个节点都在改它们，需要在前一个结果上继续合并。
  const patch: ThreadStatePatch = {};
  let artifacts = previous.artifacts;
  let toolArtifacts = previous.tool_artifacts;
  let hasPatch = false;

  for (const update of Object.values(data)) {
    if (!isRecord(update)) {
      continue;
    }

    // 中文：标题是「整体覆盖」，不需要合并，取到字符串就替换。
    if (Object.hasOwn(update, "title") && typeof update.title === "string") {
      patch.title = update.title;
      hasPatch = true;
    }

    if (Object.hasOwn(update, "artifacts")) {
      const mergedArtifacts = mergeArtifacts(artifacts, update.artifacts);
      if (mergedArtifacts !== undefined) {
        artifacts = mergedArtifacts;
        patch.artifacts = mergedArtifacts;
        hasPatch = true;
      }
    }

    if (Object.hasOwn(update, "tool_artifacts")) {
      const merged = mergeToolArtifacts(toolArtifacts, update.tool_artifacts);
      if (merged !== undefined) {
        toolArtifacts = merged;
        patch.tool_artifacts = merged;
        hasPatch = true;
      }
    }

    // 中文：merge_todos 的约定 —— null 表示「这个节点没动待办」，空数组才是「清空」。
    // 所以这里必须排除 null，否则会把已有待办误清掉。
    // DeerFlow's merge_todos treats null as "this node did not touch todos"
    // and an empty list as an explicit clear.
    if (
      Object.hasOwn(update, "todos") &&
      update.todos !== null &&
      Array.isArray(update.todos)
    ) {
      patch.todos = update.todos;
      hasPatch = true;
    }

    // 中文：goal 同理，null 表示「保留原目标」；且只有 status=active 才写入。
    // merge_goal likewise preserves the prior goal for null writes.
    if (
      Object.hasOwn(update, "goal") &&
      update.goal !== null &&
      isGoalState(update.goal)
    ) {
      patch.goal = update.goal;
      hasPatch = true;
    }
  }

  // 没有任何字段被改动时返回 undefined，让调用方跳过这次 setState，省一次重渲染。
  return hasPatch ? patch : undefined;
}
