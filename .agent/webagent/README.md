# JINX Web Dashboard

A React + Express + Vite dashboard that live-monitors a running JINX cognitive loop: live phase
progression, per-round plan and scoring, the agent's reasoning stream, IPC/RPC traffic, terminal
output, workspace files, and the working-tree diff.

The dashboard is a **read-only observer**. It never executes tools and never drives the loop — the
Python agent owns all state, and the dashboard only renders what it finds on disk.

[English](README.md) | [Русский](README_RU.md) | [中文](README_ZH.md)

---

## 1. Layout

```text
.agent/webagent/
├── server.ts              # Express API + Vite middleware (dev) or static dist/ (prod)
├── src/
│   ├── App.tsx            # Shell, tabs, polling/SSE, token prompt, session history
│   ├── server-utils.ts    # timingSafeEqual, in-memory session tracker
│   ├── diff-utils.ts      # parseDiffText()
│   ├── utils.ts           # LocalStorage sessions, folder-import parser
│   ├── types.ts           # AgentSession, SessionStatus, CodeDiff, ...
│   ├── components/        # CognitiveLoop, ThoughtStream, FileExplorer,
│   │                      # TerminalConsole, RunSummary, DiffViewer
│   ├── context/           # LanguageContext (en | ru)
│   ├── locales/           # en.ts, ru.ts, types.ts
│   └── __tests__/         # vitest suites
├── package.json           # 7 npm scripts
├── vite.config.ts         # @ alias, HMR / file-watch toggles
├── vitest.config.ts       # jsdom, globals, ./src/__tests__/setup.ts
├── metadata.json          # Marketplace descriptor (see §8)
└── dist/                  # build output: index.html, server.cjs, assets/
```

---

## 2. Requirements and quick start

* **Node.js 18+** and npm.
* The Python agent must be running separately (see the root `README.md`) for the dashboard to show
  anything other than an idle state.

```bash
cd .agent/webagent
npm install
npm run dev
```

Then open **`http://localhost:3301`**.

In a second terminal, start the agent from the repository root:

```bash
python .agent/jinx.py "your task description"
```

---

## 3. npm scripts — all seven

| Script | Command | What it does |
| :--- | :--- | :--- |
| `dev` | `tsx server.ts` | Runs `server.ts` directly through `tsx`, with Vite in middleware mode. Use this for development. |
| `build` | `vite build && esbuild server.ts ...` | Builds the client bundle into `dist/assets/`, then bundles the server to `dist/server.cjs` (CJS, `esbuild`). |
| `start` | `node dist/server.cjs` | Serves the prebuilt `dist/`. **Requires `NODE_ENV=production`** — see §4. |
| `clean` | inline `node -e` | Removes `dist/` and `server.js`. |
| `lint` | `tsc --noEmit` | Type-checks the project. There is no ESLint config. |
| `test` | `vitest run` | Runs the suite once. |
| `test:watch` | `vitest` | Runs the suite in watch mode. |

Verified locally: `npm test` reports **5 test files, 39 tests, all passing** (vitest 4.1.9).

---

## 4. Ports, and the `NODE_ENV=production` gotcha

Port selection is a single expression in `server.ts`:

```ts
const isAiStudio = process.env.AI_STUDIO === "true";
const PORT = isAiStudio ? 3000 : (process.env.PORT ? parseInt(process.env.PORT) : 3301);
```

| Situation | Port |
| :--- | :--- |
| Default local development | `3301` |
| `PORT=<n>` set | `<n>` |
| `AI_STUDIO=true` | `3000` — **`PORT` is ignored** |

**`npm run start` alone will not serve the built app.** The static-file branch is gated on
`NODE_ENV`:

```ts
if (process.env.NODE_ENV !== "production") {
  // Vite dev middleware
} else {
  // serve dist/
}
```

`npm run start` sets no such variable, so the server boots in Vite-dev mode. **This fails silently**:
the same `Server running on http://127.0.0.1:3301` line is printed and `GET /` still returns
`200 OK`, but the body is the un-built dev shell — it references `/@vite/client`, `/@react-refresh`
and `/src/main.tsx`, and contains no `assets/index-*.js` reference. Always run it as:

```bash
npm run build
NODE_ENV=production npm run start
```

The same applies in reverse: in production mode `dist/` must exist, so `build` is mandatory first.
Once set correctly, `GET /` returns the built `index.html` referencing the hashed bundle.

### Bind host

The listen address is separate from the port:

```ts
const BIND_HOST = process.env.DASHBOARD_BIND_HOST || (isAiStudio && API_TOKEN ? "0.0.0.0" : "127.0.0.1");
```

Localhost-only by default, so an unconfigured dashboard is not reachable from the network. On startup
the server logs a warning if it is bound to a non-localhost host with no `DASHBOARD_API_TOKEN` set,
because `/api/live-session` returns the contents of files from `.agent/`. Note that this warning is
written to **stderr** via `console.warn`, not to stdout — check stderr, or you will conclude it never
fired.

---

## 5. Environment variables

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `PORT` | `3301` | Listen port. Ignored when `AI_STUDIO=true`. |
| `AI_STUDIO` | *(unset)* | `true` forces port `3000`, and makes `0.0.0.0` the default bind host **when a token is also set** — AI Studio relies on port forwarding. |
| `DASHBOARD_BIND_HOST` | `127.0.0.1` | Listen address. |
| `DASHBOARD_API_TOKEN` | *(empty)* | When non-empty, the two protected endpoints require `Authorization: Bearer <token>`. |
| `NODE_ENV` | *(unset)* | Must be `production` to serve `dist/`; anything else enables Vite dev middleware. |
| `DISABLE_HMR` | *(unset)* | `true` disables both HMR and file watching in `vite.config.ts`, to cut CPU churn while the agent rewrites files. |

A `.env` file in `.agent/webagent/` is loaded automatically via `dotenv`.

---

## 6. HTTP API

| Method & path | Auth | Response |
| :--- | :--- | :--- |
| `GET /api/auth-check` | none | `{ "tokenConfigured": boolean }` — reveals *whether* a token is set, never its value. Used by the client to decide whether to prompt. |
| `GET /api/live-session` | bearer, if token set | Full session snapshot as JSON. |
| `GET /api/live-session/stream` | bearer, if token set | SSE, same payload pushed every **2 seconds**, with a `:\n\n` heartbeat first for proxy compatibility. |

The protected routes use `requireAuth`, which passes through when no token is configured, and
otherwise compares the bearer token with `crypto.timingSafeEqual`. Failures return `401` with a
`WWW-Authenticate: Bearer` header.

Because `EventSource` cannot set custom headers, the client switches transport automatically: with no
token it uses SSE; once a token is entered it falls back to 2-second `fetch` polling. The active mode
is shown in the header.

Request bodies are capped at 10 MB; oversized or malformed JSON returns `413`.

---

## 7. How session data is derived

`getLiveSessionData()` locates the agent directory by trying `<cwd>/.agent`, `<cwd>/../.agent`, and
`<cwd>/..`, accepting a directory only if it is named `.agent` or contains `JINX.yaml`.

* **Files** — every top-level file in `.agent` under 1 MB, skipping dotfiles, `node_modules` and
  `dist`, plus every `.py` file directly inside `.agent/src/`.
* **Status** — derived from `JINX.yaml` `state.exit_ready` / `state.deadlock`, refined by
  `jinx_run_state.yaml` when present: `waiting_for: llm_generate` with `tool_depth == 0` on round 1
  is `perceive`, otherwise `plan`; `tool_depth > 0` is `verify`; `waiting_for: tool_calls` is
  `execute` or `commit`.
* **Plan** — one step per `state.scores` entry, showing `pass_count`/total requirements and the prior
  failure.
* **Thoughts** — assistant text blocks from `jinx_run_state.yaml` `history` with fenced YAML/JSON
  stripped, timestamps back-dated at 15-second intervals; falls back to synthesized entries from the
  score history when the run state has no history yet.
* **IPC log** — `tool_use` blocks as `sent` and `tool_result` blocks as `received`.
* **Terminal** — `bash_exec` scripts and their results, paired by tool id.
* **Diffs** — `git diff` run at the discovered git root, with a 3-second cache, a 5-second timeout,
  a 10 MB buffer cap, and a recursive `fs.watch` that invalidates the cache on any change.
* **Session identity** — an in-memory `Map` keyed by agent directory. The first task is
  `live-session`; each new task increments to `live-session-1`, `live-session-2`, … The reported PID
  is a synthetic value in the 10000–19999 range, not a real OS process id.

**Caveat on metrics:** `promptTokens`, `completionTokens` and `estimatedCost` are hard-coded to `0`
in the live path, because the Python agent has no LLM client of its own and never sees token usage.
Only `hostname`, `os` and the synthetic PID are real.

### Client-side persistence

| localStorage key | Purpose |
| :--- | :--- |
| `jinx_sessions` | Saved run history, including imported folders. |
| `jinx_active_tab` | Currently selected tab. |
| `jinx_live_poll_active` | Whether live updates are enabled. |
| `jinx_api_token` | Bearer token, so a reload does not re-prompt. |
| `workspace_language` | `en` or `ru`. |

On a `401` the client clears `jinx_api_token` and surfaces an error asking for a valid token.

---

## 8. Interface

Five tabs: **Summary**, **Thoughts**, **Files**, **Console**, **Diffs**.

* **Summary** — the `CognitiveLoop` strip and `RunSummary`. The loop renders **seven** stages:
  Perceive, Analyze, Plan, Execute, Verify, Commit, Completed, plus `idle` and `error` states
  (error lights the last stage red). A 7-column grid on desktop, a vertical timeline on mobile.
* **Thoughts** — the agent's reasoning stream, filterable by category (monologue, question, decision,
  check, system) and by phase, with text search.
* **Files** — browsable `.agent` file contents with copy-to-clipboard.
* **Console** — two sub-tabs: terminal output and the raw IPC/RPC log.
* **Diffs** — one card per changed file with additions/deletions counts. The diff is rendered as a
  **single unified table**, one row per line, coloured by `+`/`-` prefix — not a side-by-side
  two-column comparison.

### Localization

`LanguageContext` ships exactly two locales, `en` and `ru`, defaulting to `en`. There is no `zh`
locale in the code; a Chinese-language README exists but the UI itself is English/Russian only.

### Known metadata discrepancy

`metadata.json` advertises `"name": "MachineGPT Agent Terminal"` and
`"majorCapabilities": ["MAJOR_CAPABILITY_SERVER_SIDE_GEMINI_API"]`. No Gemini dependency exists in
`package.json` and the server never calls an LLM. The name is also inconsistent with `jinx-dashboard`
in `package.json` and the `MachineGPT` references in `utils.ts`. This is a stale descriptor, not a
runtime feature.

---

## 9. Tests

```bash
npm test          # 5 files, 39 tests
npm run lint      # tsc --noEmit
```

| File | Covers |
| :--- | :--- |
| `src/__tests__/locales.test.ts` | `en` and `ru` dictionaries satisfy `TranslationDict`. |
| `src/__tests__/server-utils.test.ts` | `timingSafeEqual`, session-id sequencing. |
| `src/__tests__/components/CognitiveLoop.test.tsx` | Phase strip and status rendering. |
| `src/__tests__/components/FileExplorer.test.tsx` | File list and content display. |
| `src/__tests__/components/RunSummary.test.tsx` | Summary stats and plan rendering. |

`src/__tests__/setup.ts` registers `@testing-library/jest-dom`; `TestWrapper.tsx` supplies the
`LanguageProvider`.

---

[Back to the main JINX documentation](../../README.md)
