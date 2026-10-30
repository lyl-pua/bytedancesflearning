/**
 * Next.js 配置文件（前端总开关）。
 *
 * 【中文导读 · 初学者版】
 * 本文件最关键的只有一段：`async rewrites()` —— 反向代理规则。
 *
 * 它解决的问题：浏览器只会访问 127.0.0.1:3000（前端），但后端在 8001。
 * 如果前端直接请求 8001，就是「跨域」（CORS），cookie 也不好带。
 * 所以这里让 Next.js 服务器（Node 进程）当「中间人」：
 *     浏览器请求 /api/xxx  →  Next.js 服务器转发到  →  http://127.0.0.1:8001/api/xxx
 * 浏览器全程只跟 3000 说话，同源，没有跨域问题。
 *
 * ★ 本机踩过的坑：只要设置了环境变量 NEXT_PUBLIC_BACKEND_BASE_URL，
 *   下面 `if (!process.env.NEXT_PUBLIC_BACKEND_BASE_URL)` 就不成立，
 *   第三条 catch-all 转发规则不会被注册 → /api/v1/auth/setup-status 返回 404
 *   → 页面显示「服务暂时不可用」。所以本地运行**不要设**任何 NEXT_PUBLIC_* 。
 *
 * ● 涉及知识点：
 *   1) 反向代理（reverse proxy）：服务器替客户端去请求另一个服务。
 *   2) Next.js rewrites：只在服务端生效，浏览器看不到真实后端地址（顺带安全）。
 *   3) 环境变量开关切换部署形态：同一套代码既能「本地直连后端」，也能「走 nginx 统一入口」。
 *   4) i18n.locales：站点支持哪些语言。
 */

import { fileURLToPath } from "node:url";

/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";
import { getAllowedDevOrigins } from "./src/dev-origins.js";

/**
 * 读环境变量里的服务地址；没配就用默认值。
 * 顺便把末尾的 / 去掉，避免拼出 "http://x//api" 这种双斜杠。
 */
function getInternalServiceURL(envKey, fallbackURL) {
  const configured = process.env[envKey]?.trim();
  return configured && configured.length > 0
    ? configured.replace(/\/+$/, "")
    : fallbackURL;
}
import nextra from "nextra";

// Nextra = 文档站插件（把 markdown 渲染成文档页面），与 Agent 主流程无关
const withNextra = nextra({});

/** @type {import("next").NextConfig} */
const config = {
  // 只有显式要求时才产出 standalone 产物（给容器部署用）
  output:
    process.env.NEXT_CONFIG_BUILD_OUTPUT === "standalone"
      ? "standalone"
      : undefined,
  // 国际化：英文/中文，默认英文
  i18n: {
    locales: ["en", "zh"],
    defaultLocale: "en",
  },
  turbopack: {
    root: fileURLToPath(new URL(".", import.meta.url)),
  },
  // 关掉开发时的左下角小图标
  devIndicators: false,
  // 允许哪些来源在开发模式下访问（见 src/dev-origins.js）
  allowedDevOrigins: getAllowedDevOrigins(),
  /**
   * 反向代理规则表：数组里每条 = 一条「从哪来 → 转发到哪」。
   * 顺序很重要：越具体的规则要放越前面。
   */
  async rewrites() {
    const rewrites = [];
    // 网关地址：默认本机 8001（本地学习就是这个）
    const gatewayURL = getInternalServiceURL(
      "DEER_FLOW_INTERNAL_GATEWAY_BASE_URL",
      "http://127.0.0.1:8001",
    );

    // ① LangGraph 兼容前缀：/api/langgraph/* → 网关的 /api/*
    //    （让前端既能直连自建网关，也能换成 LangGraph Platform 云服务）
    if (!process.env.NEXT_PUBLIC_LANGGRAPH_BASE_URL) {
      rewrites.push({
        source: "/api/langgraph",
        destination: `${gatewayURL}/api`,
      });
      rewrites.push({
        source: "/api/langgraph/:path*",
        destination: `${gatewayURL}/api/:path*`,
      });
    }

    // ② 只有当「没配 NEXT_PUBLIC_BACKEND_BASE_URL」时才启用下面这些转发
    //    ★ 配了它就代表「前端和后端不同源，要直连」，这时不能再走代理
    if (!process.env.NEXT_PUBLIC_BACKEND_BASE_URL) {
      // agents / skills：有自己前缀的接口
      rewrites.push({
        source: "/api/agents",
        destination: `${gatewayURL}/api/agents`,
      });
      rewrites.push({
        source: "/api/agents/:path*",
        destination: `${gatewayURL}/api/agents/:path*`,
      });
      rewrites.push({
        source: "/api/skills",
        destination: `${gatewayURL}/api/skills`,
      });
      rewrites.push({
        source: "/api/skills/:path*",
        destination: `${gatewayURL}/api/skills/:path*`,
      });

      // Catch-all for remaining gateway API routes (models, threads, memory,
      // mcp, artifacts, uploads, suggestions, runs, etc.) that don't have
      // their own NEXT_PUBLIC_* env var toggle.
      //
      // NOTE: this must come AFTER the /api/langgraph rewrite above so that
      // LangGraph-compatible routes keep their public prefix while Gateway
      // receives its native /api/* paths.
      //
      // ③ 兜底规则：其它所有 /api/... 一律转发给网关
      //    （认证 setup-status / login、threads、runs、models 全靠这一条）
      rewrites.push({
        source: "/api/:path*",
        destination: `${gatewayURL}/api/:path*`,
      });
    }

    return rewrites;
  },
};

export default withNextra(config);
