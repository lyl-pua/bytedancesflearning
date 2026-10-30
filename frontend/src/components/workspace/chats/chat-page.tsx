"use client";

/**
 * 【中文导读 · 初学者版】
 *
 * ● 这个文件做什么：
 *   一个「对话页面」组件。它把许多零件拼在一起：顶部工具栏（标题/产物/浏览器/导出）、
 *   中间的消息列表、底部的输入框，以及目标(goal)条、待办(todo)列表、人类输入卡片等辅助面板。
 *   它本身不处理流式数据，只负责「读取状态 → 分发用户操作 → 摆布局」。
 *
 * ● 在链路里的位置：
 *   Next.js App Router 路由 /workspace/chats/[threadId] → ChatPage
 *   → useThreadStream（连 LangGraph SDK，接收流式事件并维护 thread 状态）
 *   → MessageList 渲染消息 / InputBox 收集输入并触发发送。
 *
 * ● 涉及的知识点：
 *   1. React 组件 + hooks：
 *      useState 存可变状态；useEffect 跑副作用（含清理函数）；
 *      useMemo 缓存派生结果；useRef 存「改了也不想重渲染」的值；useCallback 稳定函数引用。
 *   2. 'use client'：这是客户端组件，可以用浏览器 API（history、document 等）。
 *   3. 受控组件 / 单向数据流：输入框不自己管内容，提交走 onSubmit 回调，页面负责发请求。
 *   4. 乐观更新：用户一点发送就切走「欢迎布局」，界面先动起来，不等后端确认。
 *   5. 流式(SSE)增量渲染：thread.messages 会随事件不断变长，isLoading 表示「正在输出中」。
 *   6. Next.js App Router 导航：router.replace/push 用于换页；
 *      但线程创建成功那一刻特意用 history.replaceState，避免组件重挂载丢掉流式状态。
 *   7. 竞态防护：submissionEpochRef（提交纪元）判断异步返回后「对话是否已被切走」。
 *   8. 权限与开关：hasPermission + env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY 控制按钮是否显示。
 */

import { useQueryClient } from "@tanstack/react-query";
import { Folder } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { type PromptInputMessage } from "@/components/ai-elements/prompt-input";
import { ArtifactTrigger } from "@/components/workspace/artifacts";
import { BrowserTrigger } from "@/components/workspace/browser-view";
import { ContextUsageBadge } from "@/components/workspace/context-usage-badge";
import { ExportTrigger } from "@/components/workspace/export-trigger";
import { GoalStatus } from "@/components/workspace/goal-status";
import {
  InputBox,
  type InputBoxSubmitOptions,
} from "@/components/workspace/input-box";
import { KnowledgeScopeSelector } from "@/components/workspace/knowledge-scope-selector";
import {
  MessageList,
  MESSAGE_LIST_DEFAULT_PADDING_BOTTOM,
} from "@/components/workspace/messages";
import { ThreadContext } from "@/components/workspace/messages/context";
import { HumanInputCard } from "@/components/workspace/messages/human-input-card";
import {
  SidecarProvider,
  SidecarTrigger,
} from "@/components/workspace/sidecar";
import { ThreadArchiveStatus } from "@/components/workspace/thread-archive-status";
import { ThreadBackgroundTasks } from "@/components/workspace/thread-background-tasks";
import { ThreadExtensionActions } from "@/components/workspace/thread-extension-actions";
import { ThreadScheduledTasksButton } from "@/components/workspace/thread-scheduled-tasks-button";
import { ThreadSubagentBatches } from "@/components/workspace/thread-subagent-batches";
import { ThreadTitle } from "@/components/workspace/thread-title";
import { TodoList } from "@/components/workspace/todo-list";
import { TokenUsageIndicator } from "@/components/workspace/token-usage-indicator";
import { useActiveGoal } from "@/components/workspace/use-active-goal";
import { Welcome } from "@/components/workspace/welcome";
import { WorkspaceSidebarTrigger } from "@/components/workspace/workspace-sidebar-trigger";
import { useAuth } from "@/core/auth/AuthProvider";
import { hasPermission, PERMISSIONS } from "@/core/auth/permissions";
import {
  useBrowserControlEnabled,
  useKnowledgeBaseEnabled,
} from "@/core/features";
import { useI18n } from "@/core/i18n/hooks";
import {
  ALL_KNOWLEDGE_SCOPE,
  buildKnowledgeScopeSnapshot,
  KNOWLEDGE_SCOPE_KEY,
  type KnowledgeScopeSelection,
} from "@/core/knowledge";
import {
  buildHumanInputResponseText,
  hasOpenHumanInputRequest,
  type HumanInputRequest,
  type HumanInputResponse,
} from "@/core/messages/human-input";
import { isHiddenFromUIMessage } from "@/core/messages/utils";
import { modelTaskInput } from "@/core/models/control";
import { useModels } from "@/core/models/hooks";
import { useModelControl } from "@/core/models/use-model-control";
import { notificationBody } from "@/core/notification/body";
import { useNotification } from "@/core/notification/hooks";
import { useProject } from "@/core/projects";
import { useThreadScheduledTaskEvents } from "@/core/scheduled-tasks/events";
import { useScheduleToolResultRefresh } from "@/core/scheduled-tasks/hooks";
import { useLocalSettings, useThreadSettings } from "@/core/settings";
import { resolveThreadContext, selectThreadModel } from "@/core/settings/store";
import { createThread } from "@/core/threads/api";
import {
  useBranchThread,
  INFINITE_THREADS_QUERY_KEY_PREFIX,
  useThreadMetadata,
  useThreadStream,
  useThreadTokenUsage,
} from "@/core/threads/hooks";
import {
  selectContextUsage,
  threadTokenUsageToTokenUsage,
} from "@/core/threads/token-usage";
import { projectIdOfThread, textOfMessage } from "@/core/threads/utils";
import { env } from "@/env";
import { cn } from "@/lib/utils";

import { ChatBox } from "./chat-box";
import { useSpecificChatMode } from "./use-chat-mode";
import { useMarkOpenThreadRead } from "./use-mark-open-thread-read";
import { useThreadChat } from "./use-thread-chat";

// ★ 对话页主组件：只做「编排」——取状态、转发事件、拼布局，不直接拼装流式分片。
export default function ChatPage() {
  // 国际化文案 + 当前登录用户（后者用于权限判断）。
  const { t, locale } = useI18n();
  const { user } = useAuth();
  // 两个权限位：能否中止正在进行的运行、能否发起新的运行。
  // 前端只是「显示控制」，真正的拦截在后端。
  const canStopStreaming = hasPermission(user, PERMISSIONS.RUNS_CANCEL);
  const canCreateRuns = hasPermission(user, PERMISSIONS.RUNS_CREATE);
  // router 用来跳转页面；searchParams 用来读 URL 上的查询参数（如 ?project=xxx）。
  const router = useRouter();
  const searchParams = useSearchParams();
  // 当前对话的三个核心标记：
  // threadId 线程 id；isNewThread 后端是否「还没有」这条线程；isMock 是否为演示/假数据。
  const { threadId, setThreadId, isNewThread, setIsNewThread, isMock } =
    useThreadChat();
  // Project-scoped new chat: `/workspace/chats/new?project={id}` assigns the
  // thread to the project on the FIRST submit — primarily via an explicit
  // `POST /api/threads` pre-create with `project_id`, with the run-request
  // metadata seed as the fallback channel (the SDK's own threads.create
  // strips the reserved key, so the seed alone races the sidebar). Only
  // meaningful while the thread is still lazy (`isNewThread`); once
  // materialized the URL is replaced with the thread route and the param is
  // gone. Invalid ids are dropped by backend admission.
  // 中文：新建对话时从 URL 读取 ?project=xxx，实现「首次发送即归属某个项目」；
  // 线程一旦真正落地（isNewThread 变 false），URL 里的这个参数也就没意义了。
  const projectParam = isNewThread ? searchParams.get("project") : null;
  // `isNewThread` tracks whether the backend has the thread yet — gates the
  // SDK's history fetch (see issue #2746).  `isWelcomeMode` is the visual
  // welcome layout (centered input, hero, quick actions); we flip it to false
  // the moment the user submits so the UI animates immediately, even though
  // `isNewThread` stays true until the backend actually creates the thread.
  // 中文：isWelcomeMode 控制「欢迎态布局」（输入框居中、显示大标题和快捷入口）。
  // 它与 isNewThread 刻意分开：用户一点发送就立刻关掉欢迎态（乐观更新，界面马上有反馈），
  // 但 isNewThread 要等后端真的建好线程才变 false（否则 SDK 会去拉不存在的历史）。
  const [isWelcomeMode, setIsWelcomeMode] = useState(isNewThread);
  const queryClient = useQueryClient();
  const [settings, setSettings] = useThreadSettings(threadId);
  const modelControl = useModelControl(
    `${user?.id ?? "anonymous"}:${threadId}`,
    locale,
  );
  const prepareModelControl = modelControl.prepare;
  const [modelChoiceSummary, setModelChoiceSummary] = useState("");
  // 中文：切换对话或切换用户时清空「已选模型」提示，避免把上一个对话的选择显示过来。
  useEffect(() => setModelChoiceSummary(""), [threadId, user?.id]);
  const [localSettings, setLocalSettings] = useLocalSettings();
  const { enabled: browserControlEnabled } = useBrowserControlEnabled();
  const { tokenUsageEnabled } = useModels();
  // 中文：token 用量。新对话/演示模式没有真实线程，所以传 undefined 并关掉请求。
  const threadTokenUsage = useThreadTokenUsage(
    isNewThread || isMock ? undefined : threadId,
    { enabled: !isMock },
  );
  // 中文：线程元信息（标题、归档状态、所属项目等）。enabled 保证「只有真实存在的线程」才去请求。
  const threadMetadata = useThreadMetadata(threadId, {
    enabled: !isNewThread && !isMock,
    isMock,
  });
  // A saved thread that exists on the server is being read while open:
  // clears its unread dot (sidebar and chats list) on every device.
  // 中文：打开一个已存在的对话就算「已读」，用来清掉侧边栏的未读小红点（多端同步）。
  const markThreadRead = useMarkOpenThreadRead(threadId, {
    enabled: !isNewThread && !isMock && threadMetadata.data != null,
  });
  // Lifecycle lines of schedules created in this chat ("Paused by agent",
  // "Finished"); they stay after the task is deleted.
  const scheduledTaskEvents = useThreadScheduledTaskEvents(threadId, {
    isNewThread,
    enabled: !isMock,
  });
  const branchThread = useBranchThread();
  const backendTokenUsage = threadTokenUsageToTokenUsage(threadTokenUsage.data);
  const contextUsage = selectContextUsage(threadTokenUsage.data);
  const mountedRef = useRef(false);
  useSpecificChatMode();

  // 中文：标记「组件已完成首次挂载」。用 ref 而不是 state，因为这里不需要触发重渲染，
  // 只是给下面 InputBox 用（避免 SSR 阶段就渲染输入框导致水合不一致）。
  useEffect(() => {
    mountedRef.current = true;
  }, []);

  // Keep welcome layout in sync when navigating between threads (sidebar
  // clicks, "new chat" button).  Submitting in /chats/new flips the layout
  // via onSend below — `isNewThread` stays true until onStart, so this effect
  // is harmless during the submit transition.
  // 中文：在侧边栏点击不同对话 / 点「新建对话」时，让欢迎态布局跟着同步。
  useEffect(() => {
    setIsWelcomeMode(isNewThread);
  }, [isNewThread]);

  const { showNotification } = useNotification();
  const { scopeSelectionEnabled } = useKnowledgeBaseEnabled();
  const selectorVisible =
    scopeSelectionEnabled && env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true";
  // 中文：知识库检索范围（本次对话限定在哪些知识库/文件里搜）。
  // null = 不启用；ALL_KNOWLEDGE_SCOPE = 全部知识库。
  const [knowledgeScope, setKnowledgeScope] =
    useState<KnowledgeScopeSelection | null>(null);
  const previousConversationRef = useRef({ threadId, isNewThread });

  useEffect(() => {
    setKnowledgeScope((current) => {
      if (!selectorVisible) return null;
      return current ?? ALL_KNOWLEDGE_SCOPE;
    });
  }, [selectorVisible]);

  useEffect(() => {
    const previous = previousConversationRef.current;
    if (
      previous.threadId !== threadId ||
      previous.isNewThread !== isNewThread
    ) {
      const isNewThreadRouteReplacement = previous.isNewThread && !isNewThread;
      if (!isNewThreadRouteReplacement) {
        setKnowledgeScope(selectorVisible ? ALL_KNOWLEDGE_SCOPE : null);
      }
    }
    previousConversationRef.current = { threadId, isNewThread };
  }, [isNewThread, selectorVisible, threadId]);

  // 中文：把「知识库选择」序列化成可以随请求发送的快照对象。
  // useMemo：只有选择或开关变化时才重算，避免每次渲染都造新对象（新对象会让下游依赖失效）。
  const currentKnowledgeScopeSnapshot = useMemo(
    () =>
      selectorVisible && knowledgeScope
        ? buildKnowledgeScopeSnapshot(knowledgeScope)
        : null,
    [knowledgeScope, selectorVisible],
  );

  // ★ 核心 hook：把「服务器上的对话线程」接到 React 状态上。
  // 它内部用 LangGraph SDK 订阅流式事件：text delta 拼成消息、updates 帧合并成 values，
  // 所以 thread.messages 会随流不断变长。sendMessage / regenerateMessage 负责发起运行。
  const {
    thread,
    pendingUsageMessages,
    sendMessage,
    regenerateMessage,
    editAndRegenerateMessage,
    isUploading,
    isHistoryLoading,
    hasMoreHistory,
    loadMoreHistory,
  } = useThreadStream({
    threadId: isNewThread ? undefined : threadId,
    displayThreadId: threadId,
    context: settings.context,
    isMock,
    // onSend only animates the UI; do NOT flip `isNewThread` here — the
    // LangGraph SDK eagerly fetches /history the moment it receives a
    // thread id and assumes the thread exists on the backend (issue #2746).
    // 中文：乐观更新 —— 一点发送就关掉欢迎态，界面立刻有反馈；
    // 但刻意不改 isNewThread：线程还没真正建好，提前改会让 SDK 去拉不存在的历史。
    onSend: () => {
      setIsWelcomeMode(false);
    },
    onStart: (createdThreadId) => {
      // 中文：线程在后端创建成功 → 把地址栏换成真实线程路由，并更新本地标记。
      // 这里是全局唯一的「从新对话变成正式对话」的收口点。
      // ! Important: Never use next.js router for navigation in this case, otherwise it will cause the thread to re-mount and lose all states. Use native history API instead.
      history.replaceState(null, "", `/workspace/chats/${createdThreadId}`);
      setThreadId(createdThreadId);
      setIsNewThread(false);
    },
    // 中文：一次运行结束（用户发送，或接入了某个定时任务的运行）时回调。
    onFinish: (state) => {
      // A run in this thread ended (a send, or a joined scheduled run) while
      // it is open: it has been read.
      markThreadRead();
      // 中文：页面在后台或没有焦点时才弹系统通知，避免用户正盯着看还被打扰。
      if (document.hidden || !document.hasFocus()) {
        const lastMessage = state.messages.at(-1);
        showNotification(state.title, {
          body: notificationBody(
            lastMessage ? textOfMessage(lastMessage) : null,
          ),
        });
      }
    },
  });

  // 中文：当前线程里有没有消息（用于判断「这条对话是不是空的/不存在」）。
  const hasThreadMessages = thread.messages.length > 0;
  // A schedule_task result refreshes the header button and cards at once.
  useScheduleToolResultRefresh(isMock ? null : threadId, thread.messages);

  // 中文：兜底跳转 —— 访问一个「不存在的对话 id」，等元信息和历史都确认是空，
  // 就跳回新建对话页，避免停在一个空壳页面。
  useEffect(() => {
    if (
      !isNewThread &&
      !isMock &&
      threadMetadata.data === null &&
      !threadMetadata.isLoading &&
      !threadMetadata.isFetching &&
      !isHistoryLoading &&
      !hasMoreHistory &&
      !hasThreadMessages
    ) {
      router.replace("/workspace/chats/new");
    }
  }, [
    hasMoreHistory,
    hasThreadMessages,
    isHistoryLoading,
    isMock,
    isNewThread,
    router,
    threadMetadata.data,
    threadMetadata.isFetching,
    threadMetadata.isLoading,
  ]);

  // Born assigned: pre-create the thread row with its project so the sidebar
  // lists it under the project immediately. Idempotent server-side on
  // `thread_id`, so retrying the same first message (same `threadId` while
  // `isNewThread`) reuses the existing row instead of double-creating. This
  // is the sole membership channel — run requests never carry the project
  // key.
  //
  // Also runs before InputBox issues a `/goal <condition>` PUT (via
  // `onPrepareThread`): the goal endpoint materializes a missing thread row
  // itself, and an unassigned row would make this later idempotent create a
  // membership no-op.
  // 中文：若 URL 带 ?project=，首次发送前先把线程「预创建」并归属到该项目，
  // 这样侧边栏能立刻在项目下列出它。服务端对 thread_id 幂等，重试不会重复创建。
  const ensureProjectThread = useCallback(async () => {
    if (!projectParam) {
      return;
    }
    try {
      const created = await createThread(threadId, projectParam);
      // Keep confirmed membership available while the first metadata read is pending
      // or fails after the composer materializes this new project thread.
      queryClient.setQueryData(
        ["thread", "metadata", threadId, false],
        created,
      );
      void queryClient.invalidateQueries({
        queryKey: INFINITE_THREADS_QUERY_KEY_PREFIX,
      });
    } catch (error) {
      // Any failure (e.g. the project was deleted/archived between page load
      // and submit): do NOT submit unassigned. Reject so PromptInput keeps
      // the composer's text for a retry; the send in-flight guard is never
      // engaged on this path, so retrying works immediately.
      toast.error(t.projects.projectUnavailable);
      throw error;
    }
  }, [threadId, projectParam, queryClient, t]);

  // Submission fence. The cleanup runs when `threadId` changes (conversation
  // switch on a persisted page — sidebar navigation keeps this component
  // mounted), when the new-chat project scope changes (the sidebar "New
  // chat" link can drop `?project=` without a pathname change), or on
  // unmount. handleSubmit awaits the project pre-create before sending, and
  // a navigation during that await must not let the stale continuation
  // start a run for the abandoned conversation: its onStart would rewrite
  // the newly selected conversation's URL and its completion would clear
  // the new composer.
  // ★ 竞态防护「提交纪元」：每次切换对话/项目都会 +1。
  // 异步操作回来时对比纪元，就能知道「用户是不是已经切走了」，从而丢弃过期提交。
  const submissionEpochRef = useRef(0);
  useEffect(() => {
    return () => {
      submissionEpochRef.current += 1;
    };
  }, [threadId, projectParam]);

  // ★ 发送主流程：解析「本条消息是否要求切换模型」→ 必要时预创建线程 → 合并知识库范围
  // → 调 sendMessage 发起一次流式运行。
  // 途中每个 await 之后都核对一次提交纪元，防止用户已切换对话却还往旧对话发消息。
  const handleSubmit = useCallback(
    async (message: PromptInputMessage, options?: InputBoxSubmitOptions) => {
      const submissionEpoch = submissionEpochRef.current;
      setModelChoiceSummary("");
      const choice = await prepareModelControl(message.text);
      const modelInput = modelTaskInput(message.text, choice);
      if (submissionEpochRef.current !== submissionEpoch)
        throw new Error("thread-submission-stale");
      // 中文：runModelContext 是「本次运行使用哪个模型」的参数袋：
      // scope=task 指子任务模型，其余情况指主模型。
      const runModelContext: Record<string, unknown> = {};
      if (choice?.model_name) {
        if (choice.scope === "task") {
          if (settings.context.mode !== "ultra") {
            toast.error(
              locale.startsWith("zh")
                ? "当前未启用子任务，请先启用 Ultra 模式。"
                : "Subtasks are disabled. Enable Ultra mode first.",
            );
            throw new Error("subtasks-disabled");
          }
          runModelContext.subagent_model_name = choice.model_name;
        } else {
          runModelContext.model_name = choice.model_name;
          runModelContext.model_selection_requested = choice.model_name;
        }
      }
      // 中文：真正发出请求前，确认「项目归属」已就绪（失败会抛错，输入框内容会保留待重试）。
      await ensureProjectThread();
      // Conversation switched (or the page unmounted) while the project
      // pre-create was pending: drop the submission. Reject silently — the
      // user has already moved on, so a toast would land on the new
      // conversation — and PromptInput keeps the current composer text.
      if (submissionEpochRef.current !== submissionEpoch) {
        throw new Error("thread-submission-stale");
      }
      // 中文：用户只是「把某个模型设为本对话默认」而没写正文时，也先把空线程建出来，
      // 否则这条设置没有落点（新对话在服务端还没有 thread 行）。
      if (
        choice?.scope === "conversation" &&
        !modelInput.text &&
        message.files.length === 0 &&
        isNewThread &&
        !projectParam
      ) {
        await createThread(threadId);
        if (submissionEpochRef.current !== submissionEpoch)
          throw new Error("thread-submission-stale");
      }
      // 中文：scope=conversation 表示「以后这个对话都用它」，持久化到浏览器存储；
      // scope=turn/task 只影响本次请求，所以只写进 runModelContext。
      if (choice?.model_name) {
        if (choice.scope === "conversation")
          try {
            selectThreadModel(threadId, choice.model_name);
          } catch (error) {
            toast.error(
              locale.startsWith("zh")
                ? "无法在浏览器保存本对话的默认模型，请检查存储权限。"
                : "Unable to save this conversation's default model. Check browser storage permissions.",
            );
            throw error;
          }
        const label =
          choice.candidates.find((item) => item.name === choice.model_name)
            ?.label ?? choice.model_name;
        const scopeLabel = locale.startsWith("zh")
          ? ({
              task: "本次子任务",
              turn: "仅本次主任务",
              conversation: "本对话默认模型",
            } as const)
          : ({
              task: "This subtask",
              turn: "This turn only",
              conversation: "Conversation default",
            } as const);
        setModelChoiceSummary(`${scopeLabel[choice.scope!]} · ${label}`);
      } else setModelChoiceSummary("");
      // 中文：把「知识库检索范围」塞进 additionalKwargs，随本次运行一起发给后端。
      const scopedOptions = currentKnowledgeScopeSnapshot
        ? {
            ...options,
            additionalKwargs: {
              ...options?.additionalKwargs,
              [KNOWLEDGE_SCOPE_KEY]: currentKnowledgeScopeSnapshot,
            },
          }
        : options;
      if (
        choice?.scope === "turn" &&
        !modelInput.text &&
        message.files.length === 0
      ) {
        toast.info(
          locale.startsWith("zh")
            ? "请补充这次需要处理的任务。"
            : "Please include the task to run with this model.",
        );
        throw new Error("model-task-missing");
      }
      if (
        choice?.scope === "conversation" &&
        !modelInput.text &&
        message.files.length === 0
      ) {
        if (isNewThread) {
          history.replaceState(null, "", `/workspace/chats/${threadId}`);
          setThreadId(threadId);
          setIsNewThread(false);
        }
        options?.onSent?.();
        toast.success(
          locale.startsWith("zh")
            ? "已设置本对话的默认模型。"
            : "Conversation default model saved.",
        );
        return;
      }
      const executionOptions = {
        ...scopedOptions,
        additionalKwargs: {
          ...scopedOptions?.additionalKwargs,
          ...modelInput.additionalKwargs,
        },
      };
      // 中文：发起流式运行。有附件时把 promise 返回给输入框（让它显示上传进度），
      // 没有附件时直接 void 掉（跑完由 onFinish 回调处理即可）。
      const sendPromise = sendMessage(
        threadId,
        { ...message, text: modelInput.text },
        runModelContext,
        executionOptions,
      );
      if (message.files.length > 0) {
        return sendPromise;
      }
      void sendPromise;
    },
    [
      currentKnowledgeScopeSnapshot,
      sendMessage,
      threadId,
      ensureProjectThread,
      prepareModelControl,
      settings.context.mode,
      locale,
      isNewThread,
      projectParam,
      setThreadId,
      setIsNewThread,
    ],
  );
  // 中文：用户在「人类输入卡片」上作答后，把答案作为一条隐藏消息发回后端，
  // 让被打断的智能体从断点继续。hide_from_ui 让它不在聊天记录里显示成普通气泡。
  const handleSubmitHumanInput = useCallback(
    async (request: HumanInputRequest, response: HumanInputResponse) => {
      let sent = false;
      await sendMessage(
        threadId,
        {
          text: buildHumanInputResponseText(request, response),
          files: [],
        },
        undefined,
        {
          additionalKwargs: {
            hide_from_ui: true,
            human_input_response: response,
            ...(currentKnowledgeScopeSnapshot
              ? { [KNOWLEDGE_SCOPE_KEY]: currentKnowledgeScopeSnapshot }
              : {}),
          },
          onSent: () => {
            sent = true;
          },
        },
      );
      return sent;
    },
    [currentKnowledgeScopeSnapshot, sendMessage, threadId],
  );
  // 中文：中止当前流式运行（用户点「停止」）。
  const handleStop = useCallback(async () => {
    await thread.stop();
  }, [thread]);
  // 中文：重新生成 —— 后端会作废被替代的那批消息（superseded），再重跑一次。
  const handleRegenerate = useCallback(
    (messageId: string, supersededMessageIds: string[]) =>
      regenerateMessage(threadId, messageId, supersededMessageIds),
    [regenerateMessage, threadId],
  );
  // 中文：编辑用户消息并重新生成 —— 改文本后从这条消息开始重跑。
  const handleEditAndRegenerate = useCallback(
    (messageId: string, replacementText: string) =>
      editAndRegenerateMessage(
        threadId,
        messageId,
        replacementText,
        currentKnowledgeScopeSnapshot
          ? { [KNOWLEDGE_SCOPE_KEY]: currentKnowledgeScopeSnapshot }
          : undefined,
      ),
    [currentKnowledgeScopeSnapshot, editAndRegenerateMessage, threadId],
  );
  // 中文：从某一轮分叉出一条新对话（保留历史上下文），成功后跳到新线程。
  const handleBranchTurn = useCallback(
    async (messageId: string, messageIds: string[]) => {
      if (
        isNewThread ||
        isMock ||
        env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true"
      ) {
        return;
      }

      try {
        const response = await branchThread.mutateAsync({
          threadId,
          messageId,
          messageIds,
        });
        toast.success(t.conversation.branchCreated);
        router.push(`/workspace/chats/${response.thread_id}`);
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : t.conversation.branchFailed,
        );
      }
    },
    [branchThread, isMock, isNewThread, router, t, threadId],
  );

  // tokenUsageInlineMode：把「是否在消息里内联显示 token 用量」收敛成三态开关。
  const tokenUsageInlineMode = tokenUsageEnabled
    ? localSettings.tokenUsage.inlineMode
    : "off";
  // 中文：这几个布尔值都是「派生状态」——由已有状态算出来，不额外存一份，
  // 避免出现两份数据不一致。比如浏览器面板只有真实存在的线程才能开。
  const hasTodos = (thread.values.todos?.length ?? 0) > 0;
  const browserEnabled = !isNewThread && !isMock && browserControlEnabled;
  // 中文：当前「进行中的目标」，界面顶部的目标进度条由它驱动。
  const { activeGoal, hasGoal, goalOutcome, setLocalGoal } = useActiveGoal(
    threadId,
    thread.values.goal,
    thread.values.goal_outcome,
    thread.messages,
  );
  // 中文：是否还有「未回答的人类输入卡片」。有的话要锁住编辑/重新生成，
  // 否则用户可能绕过卡片直接把流程改掉。useMemo 避免每条消息都重扫一遍。
  const hasOpenHumanInputCard = useMemo(
    () =>
      hasOpenHumanInputRequest(
        thread.messages,
        (message) => !isHiddenFromUIMessage(message),
      ),
    [thread.messages],
  );

  // Project affiliation chip: shown once the materialized thread's metadata
  // carries `deerflow_project_id` (written by the create/move endpoints and
  // exposed here read-only).
  // 中文：线程所属项目（只读展示成一个小徽标）。新对话/演示模式没有元信息，返回 null。
  const affiliatedProjectId =
    !isNewThread && !isMock && threadMetadata.data
      ? projectIdOfThread(threadMetadata.data)
      : null;
  // A goal blocks edit-and-rerun server-side; show the pencil locked.
  // 中文：editBase 是所有「编辑类操作」的公共前置条件（线程真实存在、没有在跑、
  // 没在上传、没有待回答的输入卡片）。目标进行中时服务端禁止改写，所以再单独锁一次。
  const editBase =
    !isNewThread &&
    !isMock &&
    env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" &&
    !isUploading &&
    !thread.isLoading &&
    !branchThread.isPending &&
    !hasOpenHumanInputCard;

  // 中文：整体布局 = 顶部悬浮工具栏 + 中间消息列表 + 底部输入区。
  // 输入区在「欢迎态」时用 absolute + 负位移上移，营造居中的视觉；正式态则贴底。
  return (
    <ThreadContext.Provider value={{ thread, isMock }}>
      <SidecarProvider
        parentThreadId={threadId}
        context={settings.context}
        isMock={isMock}
      >
        <ChatBox threadId={threadId} browserEnabled={browserEnabled}>
          <div className="relative flex size-full min-h-0 justify-between">
            <header
              className={cn(
                "absolute top-0 right-0 left-0 flex h-12 shrink-0 items-center gap-2 px-2 sm:px-4",
                isWelcomeMode
                  ? "bg-background/0 z-40 backdrop-blur-none"
                  : "bg-background/80 z-30 shadow-xs backdrop-blur",
              )}
            >
              {!isMock && <WorkspaceSidebarTrigger className="md:hidden" />}
              <div className="flex min-w-0 flex-1 items-center gap-2 text-sm font-medium">
                <ThreadTitle
                  threadId={threadId}
                  thread={thread}
                  canonicalTitle={threadMetadata.data?.values?.title}
                />
                {!isNewThread &&
                  !isMock &&
                  env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" && (
                    <ThreadArchiveStatus
                      threadId={threadId}
                      metadata={threadMetadata.data?.metadata}
                    />
                  )}
                {affiliatedProjectId && (
                  <ProjectAffiliationBadge projectId={affiliatedProjectId} />
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {!isNewThread &&
                  !isMock &&
                  env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" && (
                    <ThreadBackgroundTasks threadId={threadId} />
                  )}
                {!isNewThread &&
                  !isMock &&
                  env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" && (
                    <ThreadSubagentBatches threadId={threadId} />
                  )}
                {!isNewThread && !isMock && (
                  <ThreadScheduledTasksButton threadId={threadId} />
                )}
                {tokenUsageEnabled ? (
                  <TokenUsageIndicator
                    threadId={isNewThread ? undefined : threadId}
                    backendUsage={backendTokenUsage}
                    contextUsage={contextUsage}
                    enabled={tokenUsageEnabled}
                    messages={thread.messages}
                    pendingMessages={pendingUsageMessages}
                    preferences={localSettings.tokenUsage}
                    onPreferencesChange={(preferences) =>
                      setLocalSettings("tokenUsage", preferences)
                    }
                  />
                ) : (
                  <ContextUsageBadge contextUsage={contextUsage} />
                )}
                <SidecarTrigger />
                {browserEnabled && <BrowserTrigger />}
                <ExportTrigger threadId={threadId} />
                <ThreadExtensionActions threadId={threadId} />
                <ArtifactTrigger />
              </div>
            </header>
            <main className="flex min-h-0 max-w-full grow flex-col">
              <div className="flex min-h-0 flex-1 justify-center">
                <MessageList
                  archiveDownloadsEnabled={
                    isNewThread || isMock || threadMetadata.data != null
                  }
                  className={cn("size-full", !isWelcomeMode && "pt-10")}
                  testId="main-message-list"
                  threadId={threadId}
                  thread={thread}
                  scheduledTaskEvents={scheduledTaskEvents.data}
                  enableConversationOutline
                  paddingBottom={MESSAGE_LIST_DEFAULT_PADDING_BOTTOM}
                  hasMoreHistory={hasMoreHistory}
                  loadMoreHistory={loadMoreHistory}
                  isHistoryLoading={isHistoryLoading}
                  tokenUsageInlineMode={tokenUsageInlineMode}
                  canRegenerate={
                    !isNewThread &&
                    !isMock &&
                    env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" &&
                    !isUploading &&
                    !thread.isLoading
                  }
                  onRegenerateMessage={handleRegenerate}
                  canEdit={editBase && !hasGoal}
                  editLockedByGoal={editBase && hasGoal}
                  onEditAndRegenerateMessage={handleEditAndRegenerate}
                  onSubmitHumanInput={
                    isMock || env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true"
                      ? undefined
                      : handleSubmitHumanInput
                  }
                  canBranch={
                    !isNewThread &&
                    !isMock &&
                    env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY !== "true" &&
                    !isUploading &&
                    !thread.isLoading &&
                    !branchThread.isPending
                  }
                  onBranchTurn={handleBranchTurn}
                />
              </div>
              <div
                className={cn(
                  "right-0 bottom-0 left-0 z-30 flex justify-center px-3 sm:px-4",
                  isWelcomeMode ? "absolute" : "relative shrink-0 pb-4",
                )}
              >
                <div
                  className={cn(
                    "relative w-full",
                    isWelcomeMode &&
                      "-translate-y-[calc(50vh-48px)] sm:-translate-y-[calc(50vh-96px)]",
                    isWelcomeMode
                      ? "max-w-(--container-width-sm)"
                      : "max-w-(--container-width-md)",
                  )}
                >
                  {(hasGoal || goalOutcome !== null || hasTodos) && (
                    <div
                      className={cn(
                        "right-0 left-0 z-0",
                        isWelcomeMode ? "absolute -top-4" : "relative",
                      )}
                    >
                      <div
                        className={cn(
                          "right-0 bottom-0 left-0 flex flex-col",
                          isWelcomeMode ? "absolute" : "relative",
                        )}
                      >
                        <GoalStatus
                          goal={activeGoal}
                          outcome={goalOutcome}
                          isRunning={thread.isLoading}
                          hasOpenHumanInputCard={hasOpenHumanInputCard}
                        />
                        {hasTodos && (
                          <TodoList
                            className="bg-background/5"
                            todos={thread.values.todos ?? []}
                            hidden={false}
                          />
                        )}
                      </div>
                    </div>
                  )}
                  {modelControl.request && (
                    <div className="mb-3" data-testid="model-selection-card">
                      <HumanInputCard
                        request={modelControl.request}
                        onSubmit={(response) => modelControl.choose(response)}
                      />
                    </div>
                  )}
                  {modelChoiceSummary && (
                    <p
                      className="text-muted-foreground mb-2 text-sm"
                      data-testid="model-selection-applied"
                    >
                      {modelChoiceSummary}
                    </p>
                  )}
                  {mountedRef.current ? (
                    <InputBox
                      className={cn(
                        "bg-background/5 w-full",
                        isWelcomeMode && "-translate-y-2 sm:-translate-y-4",
                      )}
                      isWelcomeMode={isWelcomeMode}
                      threadId={threadId}
                      projectId={projectParam ?? affiliatedProjectId}
                      draftThreadId={isNewThread ? "new" : threadId}
                      knowledgeScopeControl={
                        selectorVisible && knowledgeScope ? (
                          <KnowledgeScopeSelector
                            agentName="lead_agent"
                            disabled={thread.isLoading || isUploading}
                            selection={knowledgeScope}
                            onChange={setKnowledgeScope}
                          />
                        ) : undefined
                      }
                      autoFocus={isWelcomeMode}
                      status={
                        thread.error
                          ? "error"
                          : thread.isLoading
                            ? "streaming"
                            : "ready"
                      }
                      context={settings.context}
                      extraHeader={
                        isWelcomeMode &&
                        !hasGoal &&
                        !hasTodos && <Welcome mode={settings.context.mode} />
                      }
                      disabled={
                        isMock ||
                        env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" ||
                        isUploading ||
                        modelControl.busy ||
                        (!isNewThread && isHistoryLoading)
                      }
                      onContextChange={(context, options) => {
                        if (options?.automatic)
                          resolveThreadContext(threadId, context);
                        else setSettings("context", context);
                      }}
                      onGoalChange={setLocalGoal}
                      onPrepareThread={ensureProjectThread}
                      onReferenceFileAttached={() => {
                        if (!isNewThread) return;
                        history.replaceState(
                          null,
                          "",
                          `/workspace/chats/${threadId}`,
                        );
                        setThreadId(threadId);
                        setIsNewThread(false);
                      }}
                      onSubmit={handleSubmit}
                      onStop={handleStop}
                      canStopStreaming={canStopStreaming}
                      canCreateRuns={canCreateRuns}
                    />
                  ) : (
                    <div
                      aria-hidden="true"
                      className={cn(
                        "bg-background/5 h-32 w-full rounded-2xl",
                        isWelcomeMode && "-translate-y-2 sm:-translate-y-4",
                      )}
                    />
                  )}
                  {env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" && (
                    <div className="text-muted-foreground/67 w-full translate-y-12 text-center text-xs">
                      {t.common.notAvailableInDemoMode}
                    </div>
                  )}
                </div>
              </div>
            </main>
          </div>
        </ChatBox>
      </SidecarProvider>
    </ThreadContext.Provider>
  );
}

// 中文：聊天头部里的小徽标，链接到该线程所属的项目。
// 查不到项目（比如项目已被删除）时静默返回 null，不显示破链接。
/**
 * Small chip in the chat header linking to the thread's project. Hidden
 * while the project lookup is pending or when it fails (e.g. the project
 * was deleted) — an unresolvable affiliation degrades silently.
 */
function ProjectAffiliationBadge({ projectId }: { projectId: string }) {
  const { data: project } = useProject(projectId);
  if (!project) {
    return null;
  }
  return (
    <Link
      href={`/workspace/projects/${encodeURIComponent(project.id)}`}
      className="text-muted-foreground hover:text-foreground inline-flex max-w-40 shrink-0 items-center gap-1 truncate rounded-full border px-2 py-0.5 text-xs font-normal transition-colors"
    >
      <Folder className="size-3 shrink-0" />
      <span className="truncate">{project.name}</span>
    </Link>
  );
}
