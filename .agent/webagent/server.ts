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
  const header = (req.headers.authorization || "").trim();
  const scheme = "bearer ";
  if (!header.toLowerCase().startsWith(scheme)) {
    res.setHeader("WWW-Authenticate", 'Bearer realm="dashboard"'); return res.status(401).json({ error: "Unauthorized" });
  }
  const token = header.slice(scheme.length).trim();
  if (!token || !timingSafeEqual(token, API_TOKEN)) {
    res.setHeader("WWW-Authenticate", 'Bearer realm="dashboard"');
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
}

const AGENT_MARKERS = ["JINX.yaml"];

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
function getLiveSessionData() {
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
        if (stat.size < 1024 * 1024) {
          files[file] = fs.readFileSync(filepath, "utf8");
        }
      } else if (stat.isDirectory() && file === "src") {
        const srcFiles = fs.readdirSync(filepath);
        for (const sf of srcFiles) {
          const sfp = path.join(filepath, sf);
          if (fs.statSync(sfp).isFile() && sf.endsWith(".py")) {
            files[`src/${sf}`] = fs.readFileSync(sfp, "utf8");
          }
        }
      }
    } catch (e) {}
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
        thoughts,
        rpcLog,
        terminalLog,
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

// REST endpoint — returns a snapshot of the current live session.
app.get("/api/live-session", requireAuth, (req, res) => {
  try {
    const data = getLiveSessionData();
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
      const data = getLiveSessionData();
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
