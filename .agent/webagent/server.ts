/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";
import YAML from "yaml";
import { execSync } from "child_process";
import os from "os";
import { timingSafeEqual, createSessionTracker, parseDiffText } from "./src/server-utils";

dotenv.config();

const app = express();
const isAiStudio = process.env.AI_STUDIO === "true";
const PORT = isAiStudio ? 3000 : (process.env.PORT ? parseInt(process.env.PORT) : 3301);

// --- Security: bind host & API token -----------------------------------
// By default the dashboard only listens on localhost. Anyone wanting to
// expose it on the LAN/network must explicitly set DASHBOARD_BIND_HOST,
// and is strongly encouraged to also set DASHBOARD_API_TOKEN so that
// /api/live-session (which can return file contents from .agent) is not
// reachable by anyone else on the network without a token.
// AI Studio environments typically use port-forwarding, so default to
// all interfaces there; local development defaults to loopback.
const API_TOKEN = process.env.DASHBOARD_API_TOKEN || "";
const BIND_HOST = process.env.DASHBOARD_BIND_HOST || (isAiStudio && API_TOKEN ? "0.0.0.0" : "127.0.0.1");

app.use(express.json({ limit: "10mb" }));

// Require a bearer token on protected routes when DASHBOARD_API_TOKEN is set.
// If no token is configured, access still defaults to localhost-only via
// BIND_HOST, so local development keeps working without extra setup.
function requireAuth(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (!API_TOKEN) return next();
  if (!isAuthorized(req)) {
    res.setHeader("WWW-Authenticate", 'Bearer realm="dashboard"');
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
}

// Whether this request presented the configured token. With no token configured
// there is nothing to present, so the answer stays false and the caller decides:
// route access remains open, but the sensitive transcript fields stay redacted
// unless the operator has deliberately configured a token.
function isAuthorized(req: express.Request): boolean {
  if (!API_TOKEN) return false;
  const header = (req.headers.authorization || "").trim();
  const scheme = "bearer ";
  if (!header.toLowerCase().startsWith(scheme)) return false;
  const token = header.slice(scheme.length).trim();
  return !!token && timingSafeEqual(token, API_TOKEN);
}

const AGENT_MARKERS = ["JINX.yaml"];

// The File-IPC handshake the runner and the host exchange, in the order a
// transition uses them.
const IPC_FILES = ["jinx_request.yaml", "jinx_response.yaml", "jinx_run_state.yaml"];

// Cap on a file inlined into the payload. A long run's history grows with the
// tool-result cache, so a hard skip made the most interesting files vanish
// exactly when a session got busy; truncating keeps them visible.
const MAX_INLINE_FILE_BYTES = 1024 * 1024;
const TRUNCATION_NOTICE = "\n... [truncated by the dashboard at %d bytes] ...\n";

// The runner deletes the IPC files between transitions, so a file that existed
// at existsSync() can be gone by the time it is read. That is normal churn, not
// a failure, and must not fail the whole snapshot.
function isVanished(e: unknown): boolean {
  const code = (e as NodeJS.ErrnoException)?.code;
  return code === "ENOENT" || code === "ENOTDIR";
}

function readCapped(filepath: string): string {
  const stat = fs.statSync(filepath);
  if (stat.size < MAX_INLINE_FILE_BYTES) {
    return fs.readFileSync(filepath, "utf8");
  }
  const buf = Buffer.alloc(MAX_INLINE_FILE_BYTES);
  const fd = fs.openSync(filepath, "r");
  try {
    fs.readSync(fd, buf, 0, MAX_INLINE_FILE_BYTES, 0);
  } finally {
    fs.closeSync(fd);
  }
  return buf.toString("utf8") + TRUNCATION_NOTICE.replace("%d", String(stat.size));
}

// Every .py under a directory, as paths relative to it, skipping caches.
function collectSourceFiles(root: string, prefix = ""): string[] {
  const out: string[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    if (entry.name === "__pycache__" || entry.name.startsWith(".")) continue;
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      out.push(...collectSourceFiles(path.join(root, entry.name), rel));
    } else if (entry.isFile() && entry.name.endsWith(".py")) {
      out.push(rel);
    }
  }
  return out.sort();
}

function findAgentDir(): string | null {
  const pathsToTry = [
    path.join(process.cwd(), ".agent"),
    path.join(process.cwd(), "../.agent"),
    path.join(process.cwd(), ".."),
  ];

  for (const p of pathsToTry) {
    let resolved: string;
    try {
      resolved = fs.realpathSync(p);
    } catch {
      continue;
    }
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
      continue;
    }

    const basename = path.basename(resolved);
    const hasAgentMarker = AGENT_MARKERS.some((m) => fs.existsSync(path.join(resolved, m)));
    // Only accept directories that either are named `.agent` or that contain
    // the expected JINX marker files. Do NOT accept the parent directory
    // fallback unless its real basename is `.agent` or it contains markers.
    if (basename === ".agent" || hasAgentMarker) {
      return resolved;
    }
  }

  return null;
}

// Walk upwards from a starting directory until a .git folder is found.
// This is used instead of process.cwd() because the dashboard is typically
// started from .agent/webagent, where .git normally doesn't exist — only
// the actual repository root (usually one or two levels above .agent) has it.
function findGitRoot(startDir: string): string | null {
  let dir = startDir;
  for (let i = 0; i < 20; i++) {
    if (fs.existsSync(path.join(dir, ".git"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break; // reached filesystem root
    dir = parent;
  }
  return null;
}

// In-memory session tracker — assigns unique session IDs per task so the
// frontend can show each task run as a separate history entry. Uses a Map
// keyed by agentDir, so it survives multiple requests but resets on restart
// (acceptable since old sessions are preserved in the dashboard's localStorage).
const { getOrCreateSessionId, getSessionPid } = createSessionTracker();

// Simple TTL cache for git diff output to avoid execSync on every SSE tick.
// Invalidated after 3 seconds or on detected filesystem changes.
let diffCache: { diff: string; repoRoot: string; timestamp: number } | null = null;
let diffWatching = false;
let diffWatcher: fs.FSWatcher | null = null;

function getCachedGitDiff(repoRoot: string): { diff: string; error: string | null } {
  const now = Date.now();
  if (diffCache && diffCache.repoRoot === repoRoot && now - diffCache.timestamp < 3000) {
    return { diff: diffCache.diff, error: null };
  }
  try {
    const diff = execSync("git diff", {
      encoding: "utf8",
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5000,
      maxBuffer: 10 * 1024 * 1024,
    });
    diffCache = { diff, repoRoot, timestamp: now };
    // Set up a one-time recursive watcher to invalidate the cache early
    if (!diffWatching) {
      diffWatching = true;
      try {
        diffWatcher = fs.watch(repoRoot, { recursive: true }, () => { diffCache = null; });
        diffWatcher.on("error", () => { diffCache = null; });
      } catch (e) { /* fs.watch recursive not supported on all platforms */ }
    }
    return { diff, error: null };
  } catch (e: any) {
    const error = e?.signal === "SIGTERM" ? "Git diff timed out on large repo." : "Git diff failed.";
    return { diff: "", error };
  }
}

// Non-protected endpoint so the frontend can detect whether auth is needed
// without requiring a valid token. Only reveals whether a token is configured,
// never the token value itself.
app.get("/api/auth-check", (req, res) => {
  res.json({ tokenConfigured: !!API_TOKEN });
});

// Core data-fetching function shared by the REST endpoint and SSE stream.
function getLiveSessionData(authorized = false) {
  const agentDir = findAgentDir();

  if (!agentDir) {
    return {
      exists: false,
      message: "No .agent folder found at standard paths. Ensure the Python agent is running.",
      searchedPaths: [
        path.join(process.cwd(), ".agent"),
        path.join(process.cwd(), "../.agent"),
        path.join(process.cwd(), ".."),
      ],
    };
  }

  const jinxYamlPath = path.join(agentDir, "JINX.yaml");
  if (!fs.existsSync(jinxYamlPath)) {
    return {
      exists: false,
      message: "No JINX.yaml found in the detected .agent folder.",
      searchedPaths: [agentDir],
    };
  }

  const files: Record<string, string> = {};

  // Read top-level .agent files
  const dirFiles = fs.readdirSync(agentDir);
  for (const file of dirFiles) {
    if (file.startsWith(".") || file === "node_modules" || file === "dist") continue;
    const filepath = path.join(agentDir, file);
    try {
      const stat = fs.statSync(filepath);
      if (stat.isFile()) {
        // The IPC files are handled below, presence-only. They must be skipped
        // here too, not just overwritten afterwards: a sub-megabyte
        // jinx_request.yaml otherwise takes this branch and is published whole.
        if (!IPC_FILES.includes(file) && stat.size < MAX_INLINE_FILE_BYTES) {
          files[file] = fs.readFileSync(filepath, "utf8");
        }
      } else if (stat.isDirectory() && file === "src") {
        // Walk recursively. The real layout is src/jinx/*.py, so a single-level
        // scan that keeps only files found directly in src/ matches nothing and
        // silently hides the whole framework.
        for (const rel of collectSourceFiles(filepath)) {
          // Same cap as every other entry: a large .py would otherwise inflate
          // every single /api/live-session response and be re-shipped on each poll.
          // A read error unrelated to the file disappearing keeps the previous
          // behaviour: the entry is simply absent.
          try {
            files[`src/${rel}`] = readCapped(path.join(filepath, rel));
          } catch (e) {}
        }
      }
    } catch (e) {}
  }

  // The File-IPC handshake files are the live state of a run, and they only
  // exist while a transition is outstanding: the runner deletes them on success,
  // deadlock and cleanup. Listing them unconditionally keeps the panel stable
  // and makes the current phase readable, instead of entries appearing and
  // vanishing underneath the user between polls.
  //
  // Their *contents* are deliberately not published. jinx_request.yaml holds the
  // full message history, tool call parameters and the tool_result_cache;
  // jinx_run_state.yaml holds history and tool results. /api/live-session is
  // reachable without a token whenever DASHBOARD_BIND_HOST opts into network
  // access, so serving those bytes would hand any client on the network the
  // task text, file contents and tool arguments. Presence is the part the panel
  // actually needs to explain the phase.
  for (const name of IPC_FILES) {
    if (name in files) continue;
    const fp = path.join(agentDir, name);
    try {
      const stat = fs.statSync(fp);
      files[name] = `(present — ${stat.size} bytes, contents withheld)`;
    } catch (e) {
      files[name] = isVanished(e)
        ? "(not present — no transition is currently outstanding)"
        : "(unreadable)";
    }
  }

  // Check for JINX-native agent first
  const jinxRunStatePath = path.join(agentDir, "jinx_run_state.yaml");

  if (fs.existsSync(jinxYamlPath)) {
    // JINX-NATIVE COGNITIVE LOOP FLOW
    const jinxContent = fs.readFileSync(jinxYamlPath, "utf8");
    const jinxData = YAML.parse(jinxContent);
    const state = jinxData?.state || {};

    const task = state.task || "JINX Cognitive Loop Run";
    const facts = state.facts || [];
    const scores = Array.isArray(state.scores) ? state.scores : [];
    const debt = state.debt || [];
    const open = state.open || [];
    const exitReady = !!state.exit_ready;
    const deadlock = !!state.deadlock;

    let status: any = "idle";
    if (deadlock) status = "error";
    else if (exitReady) status = "completed";
    else if (fs.existsSync(jinxRunStatePath)) {
      try {
        const runStateData = YAML.parse(fs.readFileSync(jinxRunStatePath, "utf8"));
        const waitingFor = runStateData?.waiting_for;
        const toolDepth = runStateData?.tool_depth || 0;
        const rnd = runStateData?.rnd || 1;
        const hasScores = scores.length > 0;

        if (waitingFor === "llm_generate") {
          if (toolDepth > 0) {
            status = "verify";
          } else if (rnd === 1 && !hasScores) {
            status = "perceive";
          } else {
            status = "plan";
          }
        } else if (waitingFor === "tool_calls") {
          status = toolDepth > 0 ? "commit" : "execute";
        } else {
          status = "error";
        }
      } catch (e) {
        status = "error";
      }
    }

    const plan = scores.map((score: any) => {
      const totalReqs = Object.keys(score.requirements || {}).length;
      const passCount = score.pass_count || 0;
      const isLatest = score.round === scores.length;
      const stepStatus = score.all_pass
        ? "completed"
        : (isLatest && status !== "completed" && status !== "error")
          ? "running"
          : "failed";

      return {
        id: `round-${score.round}`,
        title: `Round ${score.round}: ${score.approach || "Refinement Approach"}`,
        description: `Requirements passed: ${passCount}/${totalReqs}. Prior failure: ${score.prior_failure || "None"}`,
        status: stepStatus,
      };
    });

    if (plan.length === 0) {
      plan.push({
        id: "init-step",
        title: "Round 1: Initial Cognitive Perception",
        description: "Establishing task parameters and compiling environment facts.",
        status: status === "idle" ? "pending" : "running",
      });
    }

    let thoughts: any[] = [];
    let rpcLog: any[] = [];
    let terminalLog: string[] = [];

    if (fs.existsSync(jinxRunStatePath)) {
      try {
        const runStateData = YAML.parse(fs.readFileSync(jinxRunStatePath, "utf8"));
        const history = runStateData?.history || [];
        const mtime = fs.statSync(jinxRunStatePath).mtime.getTime();

        let rpcIdx = 0;
        let thoughtIdx = 0;

        history.forEach((msg: any, msgIdx: number) => {
          const role = msg.role;
          const content = msg.content;
          const approxTime = new Date(mtime - (history.length - msgIdx) * 15000).toISOString();

          if (role === "assistant") {
            let textContent = "";
            const blocks = Array.isArray(content) ? content : (typeof content === "string" ? [{ type: "text", text: content }] : []);

            blocks.forEach((block: any) => {
              if (block.type === "text") {
                textContent += block.text || "";
              } else if (block.type === "tool_use" || block.name) {
                const toolName = block.name || block.type;
                const toolParams = block.input || block.params || {};
                const toolId = block.id || `tool-${rpcIdx}`;
                rpcLog.push({
                  id: toolId,
                  direction: "sent",
                  timestamp: approxTime,
                  method: toolName,
                  params: toolParams,
                });
                rpcIdx++;

                if (toolName === "bash_exec" && toolParams.script) {
                  terminalLog.push(`$ ${toolParams.script}`);
                }
              }
            });

            let cleanText = textContent.replace(/```(?:yaml|json)[\s\S]*?```\n*/g, "").trim();

            if (cleanText) {
              thoughts.push({
                id: `thought-${thoughtIdx++}`,
                timestamp: approxTime,
                text: cleanText,
                phase: status === "idle" ? "perceive" : status,
                category: "monologue",
              });
            }
          } else if (role === "user") {
            const blocks = Array.isArray(content) ? content : (typeof content === "string" ? [{ type: "text", text: content }] : []);

            blocks.forEach((block: any) => {
              if (block.type === "tool_result" || block.tool_use_id) {
                const toolId = block.tool_use_id;
                const toolContent = block.content || "";
                rpcLog.push({
                  id: toolId || `tool-res-${rpcIdx++}`,
                  direction: "received",
                  timestamp: approxTime,
                  method: "result",
                  result: toolContent,
                });

                const matchingCall = rpcLog.find(r => r.id === toolId && r.method === "bash_exec");
                if (matchingCall) {
                  terminalLog.push(toolContent);
                }
              }
            });
          }
        });
      } catch (e) {
        console.error("Failed to parse jinx_run_state.yaml for live logs", e);
      }
    }

    if (thoughts.length === 0) {
      scores.forEach((score: any, sIdx: number) => {
        const approxTime = new Date(Date.now() - (scores.length - sIdx) * 60000).toISOString();
        thoughts.push({
          id: `thought-hist-${score.round}-a`,
          timestamp: approxTime,
          text: `Formulating approach for Round ${score.round}: "${score.approach}". Analyzing previous failure state: "${score.prior_failure || "None"}"`,
          phase: "plan",
          category: "decision",
        });
        thoughts.push({
          id: `thought-hist-${score.round}-b`,
          timestamp: new Date(new Date(approxTime).getTime() + 15000).toISOString(),
          text: `Round ${score.round} scoring complete. Passed requirements: ${score.pass_count}/${Object.keys(score.requirements || {}).length}. All requirements satisfied: ${score.all_pass}.`,
          phase: "verify",
          category: "check",
        });
      });
    }

    if (thoughts.length === 0) {
      thoughts.push({
        id: "jinx-fall-1",
        timestamp: new Date().toISOString(),
        text: `Listening to active JINX workspace state at ${jinxYamlPath}. Initial parameters compiled: ${facts.length} scope facts detected.`,
        phase: "perceive",
        category: "system",
      });
    }

    let diffs: any[] = [];
    let diffsError: string | null = null;
    const repoRoot = findGitRoot(agentDir);
    if (repoRoot) {
      const result = getCachedGitDiff(repoRoot);
      if (result.diff && result.diff.trim()) {
        diffs = parseDiffText("workspace.diff", result.diff);
      }
      diffsError = result.error;
    }

    const sessionId = getOrCreateSessionId(agentDir, state.task || "");

    return {
      exists: true,
      path: agentDir,
      session: {
        id: sessionId,
        name: task,
        timestamp: fs.statSync(jinxYamlPath).mtime.toISOString(),
        status,
        elapsedTime: fs.existsSync(jinxRunStatePath)
          ? Math.max(0, Math.round((fs.statSync(jinxRunStatePath).mtime.getTime() - fs.statSync(jinxYamlPath).mtime.getTime()) / 1000))
          : scores.length * 60,
        stats: {
          promptTokens: 0,
          completionTokens: 0,
          estimatedCost: 0,
          pid: getSessionPid(agentDir),
          hostname: os.hostname(),
          os: process.platform,
        },
        plan,
        ...redactUnlessAuthorized(authorized),
        ...(authorized ? { thoughts, rpcLog, terminalLog } : {}),
        // `files` is filtered inside getLiveSessionData: the IPC files are
        // reported presence-only regardless of authorization, because the raw
        // bytes are the sensitive part and the panel does not need them.
        diffs,
        diffsError,
        files,
        facts,
        debt,
        open,
        exitReady,
      },
    };
  }

  return {
    exists: false,
    message: "No JINX.yaml found in .agent folder. JINX agent is not running.",
    searchedPaths: [agentDir],
  };
}

// The thoughts/rpcLog/terminalLog fields are built from the run state's
// `history`, which carries the model's message text, the parameters of every
// tool call and the full content of every tool result -- including the contents
// of whatever files the agent read. requireAuth is a no-op unless
// DASHBOARD_API_TOKEN is set, and DASHBOARD_BIND_HOST alone puts the server on
// the network, so publishing those fields to an unauthenticated client hands a
// remote caller the task text and any file the run has touched.
//
// Withholding them is the safe default: the panel still shows the phase, the
// plan, the round scores and the file list, so it stays useful. Set
// DASHBOARD_API_TOKEN to opt back in to the full transcript. The `files` map
// is filtered separately, further down, for the same reason.
function redactUnlessAuthorized(verified: boolean) {
  if (verified) return {};
  return {
    thoughts: [],
    rpcLog: [],
    terminalLog: [],
    transcriptRedacted: true,
  };
}

// REST endpoint — returns a snapshot of the current live session.
app.get("/api/live-session", requireAuth, (req, res) => {
  try {
    const data = getLiveSessionData(isAuthorized(req));
    res.json(data);
  } catch (error: any) {
    const message = error?.message || (error ? String(error) : "Failed to load live agent session");
    res.status(500).json({ error: message });
  }
});

// SSE endpoint — pushes live session data every 2 seconds.
// Falls back to the REST endpoint on the client if the browser does not
// support EventSource or when DASHBOARD_API_TOKEN requires auth headers
// that EventSource cannot set.
app.get("/api/live-session/stream", requireAuth, (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const send = () => {
    try {
      const data = getLiveSessionData(isAuthorized(req));
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch (e) {
      res.write(`data: ${JSON.stringify({ exists: false, message: String(e) })}\n\n`);
    }
  };

  res.write(":\n\n"); // initial heartbeat for proxy compatibility
  send();
  const interval = setInterval(send, 2000);

  req.on("close", () => {
    clearInterval(interval);
  });
  res.on("error", () => {
    clearInterval(interval);
  });
});

// Express error handler — returns JSON instead of HTML for parse errors
app.use((err: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  if (err.type === "entity.parse.failed" || err.type === "entity.too.large") {
    res.status(413).json({ error: "Request body too large or malformed" });
    return;
  }
  console.error("Unhandled error", err);
  res.status(500).json({ error: "Internal server error" });
});

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  const server = app.listen(PORT, BIND_HOST, () => {
    console.log(`Server running on http://${BIND_HOST}:${PORT}`);
    if (BIND_HOST !== "127.0.0.1" && BIND_HOST !== "localhost" && !API_TOKEN) {
      console.warn(
        "WARNING: dashboard is bound to a non-localhost host without DASHBOARD_API_TOKEN set. " +
        "Anyone reachable on this network can read your .agent folder contents via /api/live-session. " +
        "Set DASHBOARD_API_TOKEN to require authentication."
      );
    }
  });

  const shutdown = (signal: string) => {
    console.log(`Received ${signal}, shutting down gracefully...`);
    if (diffWatcher) try { diffWatcher.close(); } catch (e) {} server.close(() => process.exit(0));
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

startServer();
