/**
 * 【中文导读 · 初学者版】
 *
 * ● 这个文件做什么：
 *   这是一个「只在服务器上运行」的函数。它读取本次请求携带的 Cookie，
 *   再去问后端网关「当前访问者是谁」，最后返回一个带标签(tag)的结果：
 *   已登录 / 未登录 / 需要初始化 / 账号被禁用 / 后端不可用。
 *
 * ● 在链路里的位置：
 *   Next.js App Router 在服务端渲染(SSR)页面时调用它
 *   → 页面根据 tag 决定：正常渲染 / 重定向登录页 / 重定向初始化向导 / 报「后端不可用」。
 *   它是「登录态判断」的入口，夹在 UI 与后端 API 之间。
 *
 * ● 涉及的知识点：
 *   1. 服务端渲染(SSR)：这段代码在 Node 端执行，能直接读 Cookie，但拿不到 window。
 *   2. Cookie 登录态判断：凭证 access_token 存在 Cookie 中，服务端把它转发给后端校验。
 *   3. 可区分联合(tagged union)：用 tag 字段区分结果，调用方 switch 时类型安全、不会漏分支。
 *   4. 超时控制：AbortController + setTimeout，防止后端卡死把页面渲染一起拖住。
 *   5. 防御式解析：用 zod（userSchema）校验后端返回，脏数据就当「后端不可用」。
 */

import { cookies } from "next/headers";

import { isStaticWebsiteOnly } from "../static-mode";

import { AUTH_DISABLED_USER, isAuthDisabledMode } from "./auth-disabled-user";
import { AUTH_REQUEST_TIMEOUT_MS } from "./constants";
import { getGatewayConfig } from "./gateway-config";
import { STATIC_WEBSITE_USER } from "./static-user";
import { type AuthResult, parseAuthError, userSchema } from "./types";

/**
 * Fetch the authenticated user from the gateway using the request's cookies.
 * Returns a tagged AuthResult — callers use exhaustive switch, no try/catch.
 */
/**
 * 中文导读：服务端解析「当前访问者身份」的入口。
 * 判断顺序很关键：
 *   1) 纯静态演示站 → 直接当已登录（用假用户）
 *   2) 后端关闭鉴权 → 直接当已登录（用固定假用户）
 *   3) 没有 Cookie → 再问后端「系统初始化了吗」，区分「未登录」与「需要初始化」
 *   4) 有 Cookie → 带着 Cookie 问后端 /auth/me，按状态码得出结论
 * 全程不抛异常：失败一律翻译成 tag，由上层页面决定如何跳转。
 */
export async function getServerSideUser(): Promise<AuthResult> {
  // 【分支 1】纯静态演示站：没有后端，也不会有真人登录，直接返回静态用户。
  if (isStaticWebsiteOnly()) {
    return {
      tag: "authenticated",
      user: STATIC_WEBSITE_USER,
    };
  }

  // 【分支 2】关闭鉴权模式（本地开发/单机部署）：跳过 Cookie，直接用固定假用户。
  if (isAuthDisabledMode()) {
    return {
      tag: "authenticated",
      user: AUTH_DISABLED_USER,
    };
  }

  // ★ Cookie 登录态判断：登录成功后，后端会把 access_token 写进 Cookie。
  // 服务端这里能直接读到它（如果是 httpOnly Cookie，浏览器里的 JS 反而读不到）。
  // 注意 `await cookies()`：Next.js 15 起 cookies() 是异步的，必须 await。
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get("access_token");

  // 网关地址来自配置（容器里一般是内网地址）。配置缺失时先返回 config_error，
  // 而不是让后面的 fetch 抛出一个难懂的异常。
  let internalGatewayUrl: string;
  try {
    internalGatewayUrl = getGatewayConfig().internalGatewayUrl;
  } catch (err) {
    return { tag: "config_error", message: String(err) };
  }

  // 【分支 3】没有 Cookie：不知道你是谁。但可能是「系统刚部署、还没初始化」，
  // 所以要额外问一次 setup-status，否则新用户会被直接丢到登录页。
  if (!sessionCookie) {
    // No session — check whether the system has been initialised yet.
    // 超时保护：AbortController 相当于「取消令牌」，到点就 abort() 中断请求，
    // 防止后端挂住导致 SSR 一直白屏（页面渲染会等这个 await）。
    const setupController = new AbortController();
    const setupTimeout = setTimeout(
      () => setupController.abort(),
      AUTH_REQUEST_TIMEOUT_MS,
    );
    try {
      // cache: "no-store" = 不缓存，每次渲染都问真实后端（登录态绝不能被缓存）。
      const setupRes = await fetch(
        `${internalGatewayUrl}/api/v1/auth/setup-status`,
        {
          cache: "no-store",
          signal: setupController.signal,
        },
      );
      clearTimeout(setupTimeout);
      if (setupRes.ok) {
        const setupData = (await setupRes.json()) as { needs_setup?: boolean };
        // 后端说「还没初始化」：上层会重定向到初始化向导，而不是登录页。
        if (setupData.needs_setup) {
          return { tag: "system_setup_required" };
        }
      }
    } catch {
      // 问不到 setup-status（超时/网络错误）：保守地当成「未登录」，
      // 至少还能走登录页，别把用户卡在错误页死胡同里。
      clearTimeout(setupTimeout);
      // If setup-status is unreachable/times out, fall through to unauthenticated.
    }
    return { tag: "unauthenticated" };
  }

  // 【分支 4】有 Cookie：把它原样转发给后端 /auth/me 校验。
  // Cookie 就是凭证，前端不解析、不信任本地内容，一切以服务端判定为准；同样带超时。
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AUTH_REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${internalGatewayUrl}/api/v1/auth/me`, {
      headers: { Cookie: `access_token=${sessionCookie.value}` },
      cache: "no-store",
      signal: controller.signal,
    });
    clearTimeout(timeout); // Clear immediately — covers all response branches

    if (res.ok) {
      // 防御式解析：后端返回结构不合法（版本不匹配、字段缺失）时 safeParse 不抛错，
      // 只返回 success:false，避免脏数据进入 UI。
      const parsed = userSchema.safeParse(await res.json());
      if (!parsed.success) {
        console.error("[SSR auth] Malformed /auth/me response:", parsed.error);
        return { tag: "gateway_unavailable" };
      }
      // 已登录，但账号还没完成初始化（例如缺个人资料）→ needs_setup。
      if (parsed.data.needs_setup) {
        return { tag: "needs_setup", user: parsed.data };
      }
      return { tag: "authenticated", user: parsed.data };
    }
    // 401 = 未登录/凭证过期，403 = 无权限。这里再细分出「账号被禁用」，
    // 好让登录页能显示具体原因，而不是笼统的「请先登录」。
    if (res.status === 401 || res.status === 403) {
      // A disabled account gets a distinct code so the login redirect can
      // say why (#4063 gap 3); every other rejected session (expired,
      // invalid, revoked) stays a plain unauthenticated bounce.
      const code = await res
        .json()
        .then((data) => parseAuthError(data).code)
        .catch(() => null);
      if (code === "account_disabled") {
        return { tag: "account_disabled" };
      }
      return { tag: "unauthenticated" };
    }
    // 其他状态码（5xx 等）说明是网关/后端出问题，而不是用户没登录 ——
    // 必须区分开，否则会把「服务挂了」误报成「请重新登录」。
    console.error(`[SSR auth] /api/v1/auth/me responded ${res.status}`);
    return { tag: "gateway_unavailable" };
  } catch (err) {
    clearTimeout(timeout);
    console.error("[SSR auth] Failed to reach gateway:", err);
    return { tag: "gateway_unavailable" };
  }
}
