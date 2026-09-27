# JINX Web 控制面板

一个基于 React + Express + Vite 的面板，用于实时监控运行中的 JINX 认知循环：阶段推进、每轮的计划
与评分、代理的推理流、IPC/RPC 流量、终端输出、工作区文件以及工作树的 diff。

该面板是一个**只读观察者**。它从不执行工具，也不驱动循环：所有状态归 Python 代理所有，面板只是
渲染它在磁盘上找到的内容。

[English](README.md) | [Русский](README_RU.md) | [中文](README_ZH.md)

---

## 1. 目录结构

```text
.agent/webagent/
├── server.ts              # Express API + Vite 中间件（开发）或静态 dist/（生产）
├── src/
│   ├── App.tsx            # 外壳、标签页、polling/SSE、令牌提示、会话历史
│   ├── server-utils.ts    # timingSafeEqual、内存中的会话跟踪器
│   ├── diff-utils.ts      # parseDiffText()
│   ├── utils.ts           # LocalStorage 会话、文件夹导入解析器
│   ├── types.ts           # AgentSession、SessionStatus、CodeDiff……
│   ├── components/        # CognitiveLoop、ThoughtStream、FileExplorer、
│   │                      # TerminalConsole、RunSummary、DiffViewer
│   ├── context/           # LanguageContext（en | ru）
│   ├── locales/           # en.ts、ru.ts、types.ts
│   └── __tests__/         # vitest 测试套件
├── package.json           # 7 个 npm 脚本
├── vite.config.ts         # @ 别名、HMR 与文件监听开关
├── vitest.config.ts       # jsdom、globals、./src/__tests__/setup.ts
├── metadata.json          # 市场描述文件（见 §8）
└── dist/                  # 构建产物：index.html、server.cjs、assets/
```

---

## 2. 环境要求与快速开始

* **Node.js 18+** 与 npm。
* Python 代理必须单独运行（参见根目录 `README.md`），否则面板只会显示空闲状态。

```bash
cd .agent/webagent
npm install
npm run dev
```

然后打开 **`http://localhost:3301`**。

在第二个终端中，从仓库根目录启动代理：

```bash
python .agent/jinx.py "你的任务描述"
```

---

## 3. npm 脚本 —— 全部七个

| 脚本 | 命令 | 作用 |
| :--- | :--- | :--- |
| `dev` | `tsx server.ts` | 通过 `tsx` 直接运行 `server.ts`，Vite 处于中间件模式。开发时使用。 |
| `build` | `vite build && esbuild server.ts ...` | 将客户端打包到 `dist/assets/`，再把服务端打包为 `dist/server.cjs`（CJS，`esbuild`）。 |
| `start` | `node dist/server.cjs` | 提供已构建的 `dist/`。**需要 `NODE_ENV=production`** —— 见 §4。 |
| `clean` | 内联 `node -e` | 删除 `dist/` 与 `server.js`。 |
| `lint` | `tsc --noEmit` | 类型检查。项目没有 ESLint 配置。 |
| `test` | `vitest run` | 单次运行测试套件。 |
| `test:watch` | `vitest` | 以监视模式运行测试套件。 |

本地已验证：`npm test` 报告 **5 个测试文件、39 个测试，全部通过**（vitest 4.1.9）。

---

## 4. 端口，以及 `NODE_ENV=production` 这个坑

端口选择是 `server.ts` 中的单个表达式：

```ts
const isAiStudio = process.env.AI_STUDIO === "true";
const PORT = isAiStudio ? 3000 : (process.env.PORT ? parseInt(process.env.PORT) : 3301);
```

| 场景 | 端口 |
| :--- | :--- |
| 默认本地开发 | `3301` |
| 设置了 `PORT=<n>` | `<n>` |
| `AI_STUDIO=true` | `3000` —— **`PORT` 会被忽略** |

**仅执行 `npm run start` 并不会提供已构建的应用。** 静态文件分支由 `NODE_ENV` 决定：

```ts
if (process.env.NODE_ENV !== "production") {
  // Vite 开发中间件
} else {
  // 提供 dist/
}
```

`npm run start` 并未设置该变量，因此服务会以 Vite 开发模式启动。**这一失败是静默的**：仍然会打印同样
的 `Server running on http://127.0.0.1:3301`，`GET /` 也依然返回 `200 OK`，但响应体是未经构建的开发
外壳——其中引用了 `/@vite/client`、`/@react-refresh` 与 `/src/main.tsx`，且不含任何
`assets/index-*.js` 引用。请这样运行：

```bash
npm run build
NODE_ENV=production npm run start
```

反过来同样成立：在生产模式下 `dist/` 必须存在，因此 `build` 是必需的。配置正确后，`GET /` 会返回
引用了带哈希资源包的已构建 `index.html`。

### 绑定地址

监听地址与端口彼此独立：

```ts
const BIND_HOST = process.env.DASHBOARD_BIND_HOST || (isAiStudio && API_TOKEN ? "0.0.0.0" : "127.0.0.1");
```

默认仅绑定本机，因此未配置的面板无法从网络访问。启动时，若绑定到非本机地址且未设置
`DASHBOARD_API_TOKEN`，服务会输出警告——因为 `/api/live-session` 会返回 `.agent/` 中文件的内容。
请注意：该警告是通过 `console.warn` 写入 **stderr** 而非 stdout 的——应检查 stderr，否则会误以为它
从未触发。

---

## 5. 环境变量

| 变量 | 默认值 | 作用 |
| :--- | :--- | :--- |
| `PORT` | `3301` | 监听端口。当 `AI_STUDIO=true` 时被忽略。 |
| `AI_STUDIO` | *(未设置)* | `true` 会强制端口为 `3000`，并且**在同时设置了令牌时**默认绑定地址为 `0.0.0.0` —— AI Studio 依赖端口转发。 |
| `DASHBOARD_BIND_HOST` | `127.0.0.1` | 监听地址。 |
| `DASHBOARD_API_TOKEN` | *(空)* | 非空时，两个受保护的端点要求 `Authorization: Bearer <token>`。 |
| `NODE_ENV` | *(未设置)* | 必须为 `production` 才会提供 `dist/`；其他任何值都会启用 Vite 开发中间件。 |
| `DISABLE_HMR` | *(未设置)* | `true` 时在 `vite.config.ts` 中同时关闭 HMR 与文件监听，以在代理改写文件时降低 CPU 占用。 |

`.agent/webagent/` 下的 `.env` 文件会通过 `dotenv` 自动加载。

---

## 6. HTTP API

| 方法与路径 | 鉴权 | 响应 |
| :--- | :--- | :--- |
| `GET /api/auth-check` | 无 | `{ "tokenConfigured": boolean }` —— 只说明*是否*配置了令牌，绝不返回其值。客户端据此决定是否提示输入。 |
| `GET /api/live-session` | bearer（若已设置令牌） | 完整的会话快照，JSON 格式。 |
| `GET /api/live-session/stream` | bearer（若已设置令牌） | SSE，每 **2 秒**推送同一份负载，开头先发送 `:\n\n` 心跳以兼容代理。 |

受保护的路由使用 `requireAuth`：未配置令牌时直接放行，否则通过 `crypto.timingSafeEqual` 比较
bearer 令牌。失败时返回 `401` 及 `WWW-Authenticate: Bearer` 响应头。

由于 `EventSource` 无法设置自定义请求头，客户端会自动切换传输方式：没有令牌时使用 SSE；输入令牌
后改用间隔 2 秒的 `fetch` 轮询。当前模式显示在页眉。

请求体上限为 10 MB；过大或格式错误的 JSON 返回 `413`。

---

## 7. 会话数据的来源

`getLiveSessionData()` 通过依次尝试 `<cwd>/.agent`、`<cwd>/../.agent` 和 `<cwd>/..` 来定位代理
目录，仅当目录名为 `.agent` 或其中包含 `JINX.yaml` 时才接受。

* **文件** —— `.agent` 下所有小于 1 MB 的顶层文件（跳过隐藏文件、`node_modules` 与 `dist`），
  以及 `.agent/src/` 下直接包含的所有 `.py` 文件。
* **状态** —— 由 `JINX.yaml` 的 `state.exit_ready` / `state.deadlock` 推导，并在存在
  `jinx_run_state.yaml` 时加以细化：`waiting_for: llm_generate` 且 `tool_depth == 0`、处于第 1 轮时
  为 `perceive`，否则为 `plan`；`tool_depth > 0` 为 `verify`；`waiting_for: tool_calls` 为 `execute`
  或 `commit`。
* **计划** —— 每条 `state.scores` 记录对应一个步骤，显示 `pass_count`/需求总数以及此前的失败原因。
* **思路** —— 取自 `jinx_run_state.yaml` 中 `history` 的 assistant 文本块，剔除被围栏包裹的
  YAML/JSON，时间戳按 15 秒间隔向前推算；若运行状态中还没有历史，则回退为根据评分历史合成的条目。
* **IPC 日志** —— `tool_use` 块记为 `sent`，`tool_result` 块记为 `received`。
* **终端** —— `bash_exec` 脚本及其结果，按工具 id 配对。
* **差异** —— 在探测到的 git 根目录执行 `git diff`，带 3 秒缓存、5 秒超时、10 MB 缓冲区上限，
  以及会在任何变更时使缓存失效的递归 `fs.watch`。
* **会话标识** —— 以代理目录为键的内存 `Map`。首个任务为 `live-session`；每个新任务递增为
  `live-session-1`、`live-session-2`……所显示的 PID 是 10000–19999 区间内的合成值，而非真实的
  操作系统进程号。

**关于指标的说明：** 在实时路径中，`promptTokens`、`completionTokens` 与 `estimatedCost` 被硬编码为
`0`，因为 Python 代理自身没有 LLM 客户端，也永远看不到令牌用量。只有 `hostname`、`os` 和该合成
PID 是真实的。

### 客户端持久化

| localStorage 键 | 用途 |
| :--- | :--- |
| `jinx_sessions` | 已保存的运行历史，包括导入的文件夹。 |
| `jinx_active_tab` | 当前选中的标签页。 |
| `jinx_live_poll_active` | 是否启用实时更新。 |
| `jinx_api_token` | Bearer 令牌，使重新加载后无需再次提示。 |
| `workspace_language` | `en` 或 `ru`。 |

收到 `401` 时，客户端会清除 `jinx_api_token`，并提示需要提供有效令牌。

---

## 8. 界面

五个标签页：**Summary**、**Thoughts**、**Files**、**Console**、**Diffs**。

* **Summary** —— `CognitiveLoop` 条带与 `RunSummary`。循环渲染**七个**阶段：Perceive、Analyze、
  Plan、Execute、Verify、Commit、Completed，以及 `idle` 与 `error` 两种状态（出错时最后一段变红）。
  桌面端为 7 列网格，移动端为垂直时间线。
* **Thoughts** —— 代理的推理流，可按类别（monologue、question、decision、check、system）与阶段
  过滤，并支持文本搜索。
* **Files** —— 浏览 `.agent` 文件内容并复制到剪贴板。
* **Console** —— 两个子标签：终端输出与原始 IPC/RPC 日志。
* **Diffs** —— 每个改动文件一张卡片，显示增删行数。diff 以**单一合并表格**呈现，每行对应文件一
  行，按 `+`/`-` 前缀着色——并非左右两栏的并排对比。

### 本地化

`LanguageContext` 仅提供 `en` 与 `ru` 两种语言，默认 `en`。代码中不存在 `zh` locale；虽然存在中文
README，但界面本身仅有英文与俄文。

### 已知的元数据不一致

`metadata.json` 声明了 `"name": "MachineGPT Agent Terminal"` 与
`"majorCapabilities": ["MAJOR_CAPABILITY_SERVER_SIDE_GEMINI_API"]`。`package.json` 中并无 Gemini
依赖，服务端也从不调用 LLM。该名称还与 `package.json` 中的 `jinx-dashboard` 以及 `utils.ts` 中的
`MachineGPT` 引用不一致。这是一个过时的描述文件，而非运行时功能。

---

## 9. 测试

```bash
npm test          # 5 个文件，39 个测试
npm run lint      # tsc --noEmit
```

| 文件 | 覆盖内容 |
| :--- | :--- |
| `src/__tests__/locales.test.ts` | `en` 与 `ru` 字典满足 `TranslationDict`。 |
| `src/__tests__/server-utils.test.ts` | `timingSafeEqual`、会话 id 递增逻辑。 |
| `src/__tests__/components/CognitiveLoop.test.tsx` | 阶段条与状态渲染。 |
| `src/__tests__/components/FileExplorer.test.tsx` | 文件列表与内容显示。 |
| `src/__tests__/components/RunSummary.test.tsx` | 汇总统计与计划的渲染。 |

`src/__tests__/setup.ts` 注册 `@testing-library/jest-dom`；`TestWrapper.tsx` 提供
`LanguageProvider`。

---

[返回 JINX 主文档](../../README.md)
