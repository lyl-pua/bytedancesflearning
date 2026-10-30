/**
 * 【中文导读 · 初学者版】
 *
 * ● 这个文件做什么：
 *   对话页的「布局外壳」。左边是聊天内容（children），右边是可收起的侧边面板：
 *   产物(artifacts)、浏览器(browser)、伴随对话(sidecar)、消息详情(details)。
 *   它负责四件事：面板互斥（同时只开一个）、拖拽调整宽度、开合动画、移动端换成抽屉。
 *
 * ● 在链路里的位置：
 *   ChatPage 用 <ChatBox> 包住消息列表和输入框；
 *   ChatBox 再根据各个 Context 里的「谁开着」来决定渲染哪块右侧面板。
 *   它不关心对话内容，只负责「摆位置 + 动画 + 交互细节」。
 *
 * ● 涉及的知识点：
 *   1. React hooks：useState / useEffect / useMemo / useRef / useCallback。
 *   2. Context 读取：useArtifacts / useMaybeSidecar 等，让各面板共享「开/关」状态。
 *   3. 懒加载：next/dynamic 把右侧重组件延迟加载，首屏更快、更省内存。
 *   4. 可调整面板：react-resizable-panels，折叠用 collapse()、展开用 resize()。
 *   5. 响应式布局：useIsMobile 决定桌面用左右分栏、移动端用 Sheet 抽屉。
 *   6. 「渲染保留」技巧：renderedRightPanel 在关闭后延迟清空，保证收起动画期间内容还在。
 *   7. 动画性能：动画期间固定内容宽度（pinnedContentWidth），避免文字每帧重排导致抖动。
 */

import { FilesIcon, XIcon } from "lucide-react";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Layout,
  type PanelSize,
  usePanelRef,
} from "react-resizable-panels";

import { ConversationEmptyState } from "@/components/ai-elements/conversation";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  areStringArraysEqual,
  collectThreadArtifactPaths,
} from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import { env } from "@/env";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";

import { useArtifacts } from "../artifacts/context";
import { useMaybeBrowserView } from "../browser-view/context";
import {
  MessageDetailsProvider,
  useMaybeMessageDetails,
} from "../message-details/context";
import { useThread } from "../messages/context";
import { useMaybeSidecar } from "../sidecar/context";

// 中文：懒加载面板期间显示的占位内容，避免右侧突然空白一下。
function RightPanelLoading() {
  return (
    <div className="grid size-full place-items-center">
      <p role="status" className="text-muted-foreground text-sm">
        Loading panel…
      </p>
    </div>
  );
}

// 中文：next/dynamic = 代码分割 + 按需加载。下面这些面板只在用户真正打开时才下载对应代码，
// loading 组件负责加载中的过渡画面。
const ArtifactFileDetail = dynamic(
  () =>
    import("../artifacts/artifact-file-detail").then(
      (module) => module.ArtifactFileDetail,
    ),
  { loading: RightPanelLoading },
);
const ArtifactFileList = dynamic(
  () =>
    import("../artifacts/artifact-file-list").then(
      (module) => module.ArtifactFileList,
    ),
  { loading: RightPanelLoading },
);
const BrowserViewPanel = dynamic(
  () =>
    import("../browser-view/browser-view-panel").then(
      (module) => module.BrowserViewPanel,
    ),
  { loading: RightPanelLoading },
);
const SidecarPanel = dynamic(
  () =>
    import("../sidecar/sidecar-panel").then((module) => module.SidecarPanel),
  { loading: RightPanelLoading },
);

const MessageDetailsPanel = dynamic(
  () =>
    import("../message-details/message-details-panel").then(
      (module) => module.MessageDetailsPanel,
    ),
  { loading: RightPanelLoading },
);

// 中文：动画时长（毫秒）与面板默认宽度。抽成常量，保证「CSS 过渡」和「JS 定时器」同步。
const RIGHT_PANEL_ANIMATION_MS = 280;
const RIGHT_PANEL_DEFAULT_SIZE = "40%";

// 中文：右侧面板的四种「身份」。同一时刻只允许一个是 active，故用联合类型而不是多个布尔值。
type RightPanelKind = "sidecar" | "artifacts" | "browser" | "details";

// 中文：本组件的 props。threadId 用来切线程时重置状态；browserEnabled 控制浏览器面板是否可用。
interface ChatBoxProps {
  children: React.ReactNode;
  threadId: string;
  browserEnabled?: boolean;
}

// 中文：外层壳。key={threadId} 很关键 —— 切换对话时让 Provider 重新挂载，
// 自动清掉上一段对话残留的面板选择状态。
function ChatBox(props: ChatBoxProps) {
  const { setOpen: setArtifactsOpen } = useArtifacts();
  const sidecar = useMaybeSidecar();
  const browserView = useMaybeBrowserView();
  // 中文：打开「消息详情」时要关掉其他面板，实现「同时只开一个」。
  const closeOtherPanels = useCallback(() => {
    setArtifactsOpen(false);
    sidecar?.close();
    browserView?.close();
  }, [setArtifactsOpen, sidecar, browserView]);

  return (
    <MessageDetailsProvider key={props.threadId} onSelect={closeOtherPanels}>
      <ChatBoxContent {...props} />
    </MessageDetailsProvider>
  );
}

// 中文：真正干活的里层组件。React.FC 只是写法，等价于普通函数组件；
// 用 key={props.threadId} 把「线程切换」当作一次新的挂载来处理。
const ChatBoxContent: React.FC<ChatBoxProps> = ({
  children,
  threadId,
  browserEnabled = true,
}) => {
  const { t } = useI18n();
  const messageDetails = useMaybeMessageDetails();
  const detailsOpen = messageDetails?.open ?? false;
  const { thread } = useThread();
  const isMobile = useIsMobile();
  const pathname = usePathname();
  // 中文：用 ref 记住「上一次的 threadId」，好在 effect 里比较出「用户换了对话」。
  // 这类值不需要触发重渲染，所以用 useRef 而不是 useState。
  const threadIdRef = useRef(threadId);

  const {
    artifacts,
    open: artifactsOpen,
    setOpen: setArtifactsOpen,
    setArtifacts,
    select: selectArtifact,
    deselect,
    selectedArtifact,
  } = useArtifacts();
  const sidecar = useMaybeSidecar();
  const sidecarOpen = sidecar?.open ?? false;
  const browserView = useMaybeBrowserView();
  const browserViewOpen = browserEnabled && (browserView?.open ?? false);

  // 中文：静态演示模式下是否「自动选中第一个产物」，只自动一次，之后交给用户。
  const [autoSelectFirstArtifact, setAutoSelectFirstArtifact] = useState(true);
  // A reopened thread's values carry no `artifacts` (see
  // collectThreadArtifactPaths); recover them from its present_files calls.
  const stateArtifacts = thread.values.artifacts;
  // 中文：把「线程状态里的产物路径」+「消息里 present_files 调用产生的产物」合并成完整列表。
  // useMemo 缓存：只有这两个输入变了才重算，避免每次渲染都扫描全部消息。
  const threadArtifactPaths = useMemo(
    () =>
      collectThreadArtifactPaths(
        Array.isArray(stateArtifacts) ? stateArtifacts : [],
        thread.messages,
      ),
    [stateArtifacts, thread.messages],
  );
  // 中文：切换对话时取消已选产物；有产物时同步到全局产物列表。
  // 注意：空数组不能覆盖「provider 预先恢复的产物」，否则会把历史产物擦掉。
  useEffect(() => {
    const threadArtifacts =
      threadArtifactPaths.length > 0 ? [...threadArtifactPaths] : undefined;

    if (threadIdRef.current !== threadId) {
      threadIdRef.current = threadId;
      deselect();
    }

    // Update artifacts from the current thread
    // An empty initial state must not erase artifacts restored by the provider
    // before the persisted thread state has arrived.
    if (threadArtifacts && !areStringArraysEqual(artifacts, threadArtifacts)) {
      setArtifacts(threadArtifacts);
    }

    // DO NOT automatically deselect the artifact when switching threads, because the artifacts auto discovering is not work now.
    // if (
    //   selectedArtifact &&
    //   !thread.values.artifacts?.includes(selectedArtifact)
    // ) {
    //   deselect();
    // }

    if (
      env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" &&
      autoSelectFirstArtifact
    ) {
      if (threadArtifacts && threadArtifacts.length > 0) {
        setAutoSelectFirstArtifact(false);
        selectArtifact(threadArtifacts[0]!);
      }
    }
  }, [
    threadId,
    artifacts,
    autoSelectFirstArtifact,
    deselect,
    selectArtifact,
    selectedArtifact,
    setArtifacts,
    threadArtifactPaths,
  ]);

  // 中文：产物面板是否展开。规则：sidecar 打开时让位；演示模式下还要「真有产物」才展开。
  // useMemo 避免每次渲染都重算这个组合判断。
  const artifactPanelOpen = useMemo(() => {
    if (sidecarOpen) {
      return false;
    }
    if (env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true") {
      return artifactsOpen && artifacts?.length > 0;
    }
    return artifactsOpen;
  }, [artifactsOpen, artifacts, sidecarOpen]);

  // 中文：按「优先级」算当前该显示哪个面板：sidecar > 浏览器 > 产物 > 消息详情。
  // 用一条三元链写出优先级，保证任何时刻只有一个 activeRightPanel。
  const activeRightPanel: RightPanelKind | null = sidecarOpen
    ? "sidecar"
    : browserViewOpen
      ? "browser"
      : artifactPanelOpen
        ? "artifacts"
        : detailsOpen
          ? "details"
          : null;
  const rightPanelOpen = activeRightPanel !== null;
  const [renderedRightPanel, setRenderedRightPanel] =
    useState<RightPanelKind | null>(activeRightPanel);

  // 中文：把路径里的非法字符替换掉，作为面板组件的唯一 id 前缀。
  // 因为同一页面可能同时存在多个 PanelGroup，id 必须唯一，否则会互相串。
  const resizableIdBase = useMemo(() => {
    return pathname.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  }, [pathname]);

  const sidePanelRef = usePanelRef();
  // 中文：记住用户手动拖出来的宽度，下次打开时按这个宽度还回来（而不是回到默认 40%）。
  // Width the panel reopens at: the last size the user dragged it to.
  const openSizeRef = useRef(RIGHT_PANEL_DEFAULT_SIZE);
  // 中文：动画期间把内容宽度「钉住」为最终宽度（cqw 相对容器宽），然后裁切溢出部分，
  // 避免文字每一帧都重新换行——那样会导致消息列表反复执行「滚到底部」，视觉上很抖。
  // While the panel width animates, the content is held at its final width and
  // clipped instead of reflowing every frame — a reflowing message list keeps
  // re-running its scroll-to-bottom (pinned by the sidecar scroll regression
  // test), and re-wrapping text mid-animation looks unsettled. Expressed in
  // `cqw` against the group so it tracks the final width even if the group
  // itself resizes during the animation.
  const [pinnedContentWidth, setPinnedContentWidth] = useState<string | null>(
    null,
  );
  // 中文：只在「开/关」时启用尺寸过渡；拖拽时若还开过渡，每一帧都会被插值，手感发飘。
  // Size transitions belong to open/close only: leaving them on during a drag
  // would interpolate every pointer frame and make the handle feel laggy.
  const [animatingRightPanel, setAnimatingRightPanel] = useState(false);
  const rightPanelOpenRef = useRef(rightPanelOpen);
  // Read once: `defaultSize` only applies on mount, the effect below owns every
  // later open/close.
  const [initialRightPanelSize] = useState(() =>
    rightPanelOpen ? RIGHT_PANEL_DEFAULT_SIZE : "0%",
  );

  // 中文：拖拽过程中持续记录宽度，只记录「面板是打开的且宽度 > 0」的帧，
  // 否则会把收起时的 0% 也记下来。
  const handleSidePanelResize = useCallback((size: PanelSize) => {
    if (!rightPanelOpenRef.current || size.asPercentage <= 0) {
      return;
    }
    openSizeRef.current = `${size.asPercentage}%`;
  }, []);

  // 中文：用户把分隔条一直拖到 0%（想收起面板）时，等「松手」再真正关闭。
  // 如果在第一帧 0% 就立刻 close，会打断「拖到边又往回拉」的连续手势。
  const handlePanelGroupLayoutChanged = useCallback(
    (layout: Layout) => {
      if (
        !rightPanelOpenRef.current ||
        layout[`${resizableIdBase}-side`] !== 0
      ) {
        return;
      }

      // Finalize a drag-collapse only after the pointer is released. Closing
      // from onResize at the first 0% frame would break a continuous gesture
      // that reaches the edge and then reverses before release.
      if (activeRightPanel === "sidecar") {
        sidecar?.close();
      } else if (activeRightPanel === "browser") {
        browserView?.close();
      } else if (activeRightPanel === "details") {
        messageDetails?.close();
      } else if (activeRightPanel === "artifacts") {
        setArtifactsOpen(false);
      }
    },
    [
      activeRightPanel,
      browserView,
      messageDetails,
      resizableIdBase,
      setArtifactsOpen,
      sidecar,
    ],
  );

  // ★ 开/关动画的核心 effect：
  // 1) 先用 ref 判断「开合状态真的变了」，避免无关重渲染触发动画；
  // 2) 关闭前先量下当前宽度（关闭动画中采到的是缩小的值，会把下次打开弄得很窄）；
  // 3) 把内容宽度钉住 → 调 resize()（不是 expand()，否则库会按最小宽度恢复）或 collapse()；
  // 4) 用定时器在动画时长后解除「动画中」标记，并返回清理函数防止提前卸载。
  useEffect(() => {
    if (rightPanelOpenRef.current === rightPanelOpen) {
      return;
    }
    rightPanelOpenRef.current = rightPanelOpen;

    setAnimatingRightPanel(true);
    if (!rightPanelOpen) {
      // Remember the width now: the closing animation reports shrinking sizes,
      // so sampling those would reopen the panel a few pixels wide.
      const size = sidePanelRef.current?.getSize().asPercentage ?? 0;
      if (size > 0) {
        openSizeRef.current = `${size}%`;
      }
    }

    const openPercentage = Number.parseFloat(openSizeRef.current);
    setPinnedContentWidth(
      Number.isFinite(openPercentage) ? `${openPercentage}cqw` : null,
    );

    // resize() rather than expand(): the library expands to `minSize` until it
    // has recorded a size of its own, which would reopen narrower than before.
    if (rightPanelOpen) {
      sidePanelRef.current?.resize(openSizeRef.current);
    } else {
      sidePanelRef.current?.collapse();
    }

    const timeout = window.setTimeout(() => {
      setAnimatingRightPanel(false);
      setPinnedContentWidth(null);
    }, RIGHT_PANEL_ANIMATION_MS);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [rightPanelOpen, sidePanelRef]);

  // 中文：关闭时不要立刻清空内容 —— 延迟到动画结束再清，否则收起过程中会先白一下。
  useEffect(() => {
    if (activeRightPanel) {
      setRenderedRightPanel(activeRightPanel);
      return;
    }

    const timeout = window.setTimeout(() => {
      setRenderedRightPanel(null);
    }, RIGHT_PANEL_ANIMATION_MS);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [activeRightPanel]);

  // 中文：下面这一组 effect 处理「面板互斥」：同时打开两个时，让优先级高的留下。
  useEffect(() => {
    if (sidecarOpen && artifactsOpen) {
      setArtifactsOpen(false);
    }
  }, [artifactsOpen, setArtifactsOpen, sidecarOpen]);

  useEffect(() => {
    if (!browserEnabled && browserView?.open) {
      browserView.close();
    }
  }, [browserEnabled, browserView]);

  useEffect(() => {
    if (detailsOpen && (sidecarOpen || browserViewOpen || artifactPanelOpen)) {
      messageDetails?.close({ restoreFocus: false });
    }
  }, [
    detailsOpen,
    sidecarOpen,
    browserViewOpen,
    artifactPanelOpen,
    messageDetails,
  ]);

  // 中文：提前算好右侧要渲染什么内容。useMemo 让「只有依赖变化时才重建元素」，
  // 避免父组件无关重渲染导致面板频繁重新挂载（比如浏览器 iframe 被重载）。
  const rightPanelContent = useMemo(() => {
    if (renderedRightPanel === "details") {
      return <MessageDetailsPanel />;
    }
    if (renderedRightPanel === "browser") {
      return <BrowserViewPanel threadId={threadId} className="size-full" />;
    }
    if (renderedRightPanel === "sidecar") {
      return <SidecarPanel />;
    }
    if (renderedRightPanel === "artifacts" && selectedArtifact) {
      return (
        <ArtifactFileDetail
          className="size-full"
          filepath={selectedArtifact}
          threadId={threadId}
        />
      );
    }
    if (renderedRightPanel === "artifacts") {
      return (
        <div className="relative flex size-full justify-center">
          <div className="absolute top-1 right-1 z-30">
            <Button
              size="icon-sm"
              variant="ghost"
              onClick={() => {
                setArtifactsOpen(false);
              }}
            >
              <XIcon />
            </Button>
          </div>
          {artifacts.length === 0 ? (
            <ConversationEmptyState
              icon={<FilesIcon />}
              title="No artifact selected"
              description="Select an artifact to view its details"
            />
          ) : (
            <div className="flex size-full max-w-(--container-width-sm) flex-col justify-center p-4 pt-8">
              <header className="shrink-0">
                <h2 className="text-lg font-medium">Artifacts</h2>
              </header>
              <ScrollArea className="min-h-0 grow">
                <ArtifactFileList
                  className="max-w-(--container-width-sm) p-4 pt-12"
                  files={artifacts}
                  threadId={threadId}
                />
              </ScrollArea>
            </div>
          )}
        </div>
      );
    }
    return null;
  }, [
    renderedRightPanel,
    selectedArtifact,
    threadId,
    artifacts,
    setArtifactsOpen,
  ]);

  // 中文：移动端分支 —— 不做左右分栏，而是用 Sheet（底部/侧边抽屉）覆盖显示面板。
  // 桌面端的拖拽宽度逻辑在手机上没有意义，所以整块走另一套实现。
  if (isMobile) {
    return (
      <>
        <div className="relative size-full min-w-0">{children}</div>
        <Sheet
          open={rightPanelOpen}
          onOpenChange={(open) => {
            if (open) {
              return;
            }
            if (detailsOpen) {
              messageDetails?.close();
            }
            if (sidecarOpen) {
              sidecar?.close();
            }
            if (browserViewOpen) {
              browserView?.close();
            }
            if (artifactsOpen) {
              setArtifactsOpen(false);
            }
          }}
        >
          <SheetContent
            className="w-[calc(100vw-1rem)] max-w-none gap-0 p-0 sm:max-w-md [&>button]:hidden"
            side="right"
          >
            <SheetHeader className="sr-only">
              <SheetTitle>
                {renderedRightPanel === "sidecar"
                  ? "Sidecar"
                  : renderedRightPanel === "browser"
                    ? "Browser"
                    : renderedRightPanel === "details"
                      ? (messageDetails?.selectedDetail?.title ?? t.common.more)
                      : "Artifacts"}
              </SheetTitle>
              <SheetDescription>
                Browse the side panel for this conversation.
              </SheetDescription>
            </SheetHeader>
            <div className="min-h-0 flex-1 p-3 pt-10">{rightPanelContent}</div>
          </SheetContent>
        </Sheet>
      </>
    );
  }

  // 中文：桌面端分支 —— 用 ResizablePanelGroup 做左右两栏：左栏是聊天，右栏是可拖拽/可折叠的面板。
  // 注意：动画相关的 class 必须挂在库自己的 [data-panel] 元素上，所以用 [&>[data-panel]] 选择器。
  return (
    <ResizablePanelGroup
      id={`${resizableIdBase}-group`}
      orientation="horizontal"
      onLayoutChanged={handlePanelGroupLayoutChanged}
      className={cn(
        "[container-type:inline-size] size-full min-h-0",
        // The sized flex item is the library's own `[data-panel]` element, not
        // the child `className` lands on, so the open/close transition has to be
        // addressed from here.
        animatingRightPanel &&
          "[&>[data-panel]]:transition-[flex-grow] [&>[data-panel]]:duration-[280ms] [&>[data-panel]]:ease-out motion-reduce:[&>[data-panel]]:transition-none",
      )}
    >
      <ResizablePanel
        id={`${resizableIdBase}-chat`}
        minSize="30%"
        className="relative min-h-0 min-w-0"
      >
        <div className="relative size-full min-h-0 min-w-0" id="chat">
          {children}
        </div>
      </ResizablePanel>
      <ResizableHandle
        id={`${resizableIdBase}-separator`}
        withHandle
        disabled={!rightPanelOpen}
        className={cn(
          "opacity-33 transition-opacity duration-200 ease-out hover:opacity-100 motion-reduce:transition-none",
          !rightPanelOpen && "pointer-events-none opacity-0",
        )}
      />
      <ResizablePanel
        id={`${resizableIdBase}-side`}
        panelRef={sidePanelRef}
        collapsible
        collapsedSize="0%"
        defaultSize={initialRightPanelSize}
        minSize="20%"
        onResize={handleSidePanelResize}
        className="min-h-0 min-w-0"
      >
        <aside
          aria-hidden={!rightPanelOpen}
          className={cn(
            "size-full min-h-0 min-w-0 overflow-hidden transition-opacity duration-[280ms] ease-out motion-reduce:transition-none",
            rightPanelOpen ? "opacity-100" : "pointer-events-none opacity-0",
          )}
          id="artifacts"
        >
          <div
            className={cn(
              "ml-auto h-full",
              renderedRightPanel === "artifacts" ? "p-4" : "p-0",
            )}
            style={
              pinnedContentWidth === null
                ? undefined
                : { width: pinnedContentWidth }
            }
          >
            {rightPanelContent}
          </div>
        </aside>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export { ChatBox };
