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
import crypto from "crypto";
import os from "os";

dotenv.config();

const app = express();
const isAiStudio = process.cwd() === "/" || fs.existsSync("/metadata.json");
const PORT = isAiStudio ? 3000 : (process.env.PORT ? parseInt(process.env.PORT) : 3301);

// --- Security: bind host & API token -----------------------------------
// By default the dashboard only listens on localhost. Anyone wanting to
// expose it on the LAN/network must explicitly set DASHBOARD_BIND_HOST,
// and is strongly encouraged to also set DASHBOARD_API_TOKEN so that
// /api/live-session (which can return file contents from .agent) is not
// reachable by anyone else on the network without a token.
// AI Studio environments typically use port-forwarding, so default to
// all interfaces there; local development defaults to loopback.
const BIND_HOST = process.env.DASHBOARD_BIND_HOST || (isAiStudio ? "0.0.0.0" : "127.0.0.1");
const API_TOKEN = process.env.DASHBOARD_API_TOKEN || "";

app.use(express.json({ limit: "50mb" }));

// Require a bearer token on protected routes when DASHBOARD_API_TOKEN is set.
// If no token is configured, access still defaults to localhost-only via
// BIND_HOST, so local development keeps working without extra setup.
function timingSafeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    crypto.timingSafeEqual(bufA, bufA);
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}

function requireAuth(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (!API_TOKEN) return next();
  const header = (req.headers.authorization || "").trim();
  const scheme = "bearer ";
  const idx = header.toLowerCase().indexOf(scheme);
  if (idx === -1) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  const token = header.slice(idx + scheme.length).trim();
  if (!token || !timingSafeEqual(token, API_TOKEN)) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
}

// Helper to find the .agent folder
function findAgentDir(): string | null {
  const pathsToTry = [
    path.join(process.cwd(), ".agent"),
    path.join(process.cwd(), "../.agent"),
    path.join(process.cwd(), ".."), // if inside webagent, process.cwd() is .../.agent/webagent, so '..' is the .agent folder
  ];

  for (const p of pathsToTry) {
    if (fs.existsSync(p) && fs.statSync(p).isDirectory()) {
      // Confirm it looks like an agent folder (has JINX.yaml, state.json, plan.json, or other typical files)
      if (
        fs.existsSync(path.join(p, "JINX.yaml")) ||
        fs.existsSync(path.join(p, "state.json")) ||
        fs.existsSync(path.join(p, "plan.json")) ||
        fs.existsSync(path.join(p, "thoughts.json")) ||
        fs.existsSync(path.join(p, "thought.json")) ||
        fs.existsSync(path.join(p, "terminal.log")) ||
        fs.existsSync(path.join(p, "stdout.log"))
      ) {
        return p;
      }
    }
  }

  // Fallback to first path if it exists
  if (fs.existsSync(pathsToTry[0])) {
    return pathsToTry[0];
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
const sessionTracker = new Map<string, { currentTask: string; seq: number }>();

function getOrCreateSessionId(agentDir: string, task: string): string {
  let tracker = sessionTracker.get(agentDir);
  if (!tracker) {
    tracker = { currentTask: "", seq: 0 };
    sessionTracker.set(agentDir, tracker);
  }

  if (task && tracker.currentTask !== task) {
    tracker.seq++;
    tracker.currentTask = task;
  }

  return tracker.seq === 0 ? "live-session" : `live-session-${tracker.seq}`;
}

// Function to parse diff text
function parseDiffText(filename: string, text: string) {
  const diffs: any[] = [];
  const lines = text.split("\n");
  let currentFile = filename;
  let currentDiffLines: string[] = [];
  let additions = 0;
  let deletions = 0;

  for (const line of lines) {
    if (line.startsWith("diff --git")) {
      if (currentDiffLines.length > 0) {
        diffs.push({
          filepath: currentFile,
          filename: currentFile.split("/").pop() || currentFile,
          additions,
          deletions,
          diffText: currentDiffLines.join("\n"),
        });
      }
      currentDiffLines = [];
      additions = 0;
      deletions = 0;
      const match = line.match(/b\/(.+)$/);
      if (match) currentFile = match[1];
    }
    currentDiffLines.push(line);
    if (line.startsWith("+") && !line.startsWith("+++")) additions++;
    if (line.startsWith("-") && !line.startsWith("---")) deletions++;
  }

  if (currentDiffLines.length > 0) {
    diffs.push({
      filepath: currentFile,
      filename: currentFile.split("/").pop() || currentFile,
      additions,
      deletions,
      diffText: currentDiffLines.join("\n"),
    });
  }

  if (diffs.length === 0 && text.trim().length > 0) {
    return [{
      filepath: filename,
      filename: filename.split("/").pop() || filename,
      additions: text.split("\n").filter(l => l.startsWith("+")).length,
      deletions: text.split("\n").filter(l => l.startsWith("-")).length,
      diffText: text,
    }];
  }

  return diffs;
}

// Non-protected endpoint so the frontend can detect whether auth is needed
// without requiring a valid token. Only reveals whether a token is configured,
// never the token value itself.
app.get("/api/auth-check", (req, res) => {
  res.json({ tokenConfigured: !!API_TOKEN });
});

// REST Endpoint to fetch real live agent logs from .agent directory
app.get("/api/live-session", requireAuth, (req, res) => {
  const agentDir = findAgentDir();

  if (!agentDir) {
    return res.json({
      exists: false,
      message: "No .agent folder found at standard paths. Ensure the Python agent is running.",
      searchedPaths: [
        path.join(process.cwd(), ".agent"),
        path.join(process.cwd(), "../.agent"),
        path.join(process.cwd(), ".."),
      ],
    });
  }

  try {
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
          // List files in src
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
    const jinxYamlPath = path.join(agentDir, "JINX.yaml");
    const jinxRunStatePath = path.join(agentDir, "jinx_run_state.yaml");

    if (fs.existsSync(jinxYamlPath)) {
      // JINX-NATIVE COGNITIVE LOOP FLOW
      const jinxContent = fs.readFileSync(jinxYamlPath, "utf8");
      const jinxData = YAML.parse(jinxContent);
      const state = jinxData?.state || {};

      const task = state.task || "JINX Cognitive Loop Run";
      const facts = state.facts || [];
      const scores = state.scores || [];
      const debt = state.debt || [];
      const open = state.open || [];
      const exitReady = !!state.exit_ready;
      const deadlock = !!state.deadlock;

      // Determine Status
      let status: any = "idle";
      const anyAllPass = scores.some((s: any) => s.all_pass === true);
      if (exitReady && anyAllPass) status = "completed";
      else if (deadlock) status = "error";
      else if (exitReady) status = "completed";
      else if (fs.existsSync(jinxRunStatePath)) {
        try {
          const runStateData = YAML.parse(fs.readFileSync(jinxRunStatePath, "utf8"));
          if (runStateData?.waiting_for === "tool_calls") {
            status = "execute";
          } else {
            status = "analyze";
          }
        } catch (e) {
          status = "execute";
        }
      }

      // Map Plan Steps from Round Scores
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

      // If no rounds executed yet, create a placeholder step
      if (plan.length === 0) {
        plan.push({
          id: "init-step",
          title: "Round 1: Initial Cognitive Perception",
          description: "Establishing task parameters and compiling environment facts.",
          status: status === "idle" ? "pending" : "running",
        });
      }

      // Extract Thoughts and RPC Log from jinx_run_state.yaml (active conversation history)
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
            const approxTime = new Date(mtime - (history.length - msgIdx) * 12000).toISOString();

            if (role === "assistant") {
              let textContent = "";
              const blocks = Array.isArray(content) ? content : (typeof content === "string" ? [{ type: "text", text: content }] : []);

              blocks.forEach((block: any) => {
                if (block.type === "text") {
                  textContent += block.text || "";
                } else if (block.type === "tool_use" || block.name) {
                  // RPC Message: Tool Call
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

                  if (toolName === "bash_exec" && toolParams.script) {
                    terminalLog.push(`$ ${toolParams.script}`);
                  }
                }
              });

              // Clean state block from monologue text if present
              let cleanText = textContent;
              const stateBlockMatch = textContent.match(/```(?:yaml|json)[\s\S]*?```/);
              if (stateBlockMatch) {
                cleanText = textContent.replace(stateBlockMatch[0], "").trim();
              }

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

                  // If this was a bash execution, pipe its stdout to the terminalLog
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

      // Generate historical thoughts if run_state has no thoughts yet
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

      // Fallback if still empty
      if (thoughts.length === 0) {
        thoughts.push({
          id: "jinx-fall-1",
          timestamp: new Date().toISOString(),
          text: `Listening to active JINX workspace state at ${jinxYamlPath}. Initial parameters compiled: ${facts.length} scope facts detected.`,
          phase: "perceive",
          category: "system",
        });
      }

      // Gather real live workspace diff using git diff.
      // Resolved relative to the actual repository root (walked up from
      // agentDir), not process.cwd() — the server is typically started from
      // .agent/webagent, which usually has no .git of its own.
      let diffs: any[] = [];
      let diffsError: string | null = null;
      const repoRoot = findGitRoot(agentDir);
      if (repoRoot) {
        try {
          const gitDiff = execSync("git diff", {
            encoding: "utf8",
            cwd: repoRoot,
            stdio: ["ignore", "pipe", "ignore"],
            timeout: 5000,
            maxBuffer: 10 * 1024 * 1024,
          });
          if (gitDiff && gitDiff.trim()) {
            diffs = parseDiffText("workspace.diff", gitDiff);
          }
        } catch (e: any) {
          diffsError = e?.message?.includes("timed out") ? "Git diff timed out on large repo." : "Git diff failed.";
        }
      }

      const sessionId = getOrCreateSessionId(agentDir, task);

      return res.json({
        exists: true,
        path: agentDir,
        session: {
          id: sessionId,
          name: task,
          timestamp: fs.statSync(jinxYamlPath).mtime.toISOString(),
          status,
          elapsedTime: scores.length * 60, // Estimated duration based on rounds
          stats: {
            promptTokens: 0,
            completionTokens: 0,
            estimatedCost: 0,
            pid: process.pid,
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
        },
      });
    }

    // LEGACY RETRO-COMPATIBILITY FALLBACK FLOW
    let status = "idle";
    let pid = 0;
    const errors: string[] = [];

    // Parse state.json
    if (files["state.json"]) {
      try {
        const stateObj = JSON.parse(files["state.json"]);
        status = (stateObj.phase || stateObj.status || "idle").toLowerCase();
        pid = stateObj.active_pid || stateObj.pid || 0;
        if (stateObj.errors && Array.isArray(stateObj.errors)) {
          errors.push(...stateObj.errors);
        }
      } catch (e) {
        console.error("Failed to parse state.json", e);
      }
    }

    // Parse plan.json
    let plan: any[] = [];
    if (files["plan.json"]) {
      try {
        const parsed = JSON.parse(files["plan.json"]);
        const rawList = Array.isArray(parsed) ? parsed : (parsed.steps || parsed.plan || []);
        plan = rawList.map((p: any, idx: number) => {
          if (typeof p === "string") {
            return { id: `step-${idx}`, title: p, description: "", status: "pending" };
          }
          return {
            id: p.id || `step-${idx}`,
            title: p.title || p.name || "Untitled Step",
            description: p.description || p.desc || "",
            status: p.status || "pending",
          };
        });
      } catch (e) {
        console.error("Failed to parse plan.json", e);
      }
    }

    // Parse thoughts.json or thought.json
    let thoughts: any[] = [];
    const thoughtsKey = files["thoughts.json"] ? "thoughts.json" : (files["thought.json"] ? "thought.json" : null);
    if (thoughtsKey && files[thoughtsKey]) {
      try {
        const parsed = JSON.parse(files[thoughtsKey]);
        const rawList = Array.isArray(parsed) ? parsed : [];
        thoughts = rawList.map((t: any, idx: number) => {
          return {
            id: t.id || `thought-${idx}`,
            timestamp: t.timestamp || new Date().toISOString(),
            text: t.text || t.thought || (typeof t === "string" ? t : ""),
            phase: (t.phase || "execute").toLowerCase(),
            category: t.category || "monologue",
          };
        });
      } catch (e) {
        console.error("Failed to parse thoughts", e);
      }
    }

    // Parse rpc.log / ipc.log
    let rpcLog: any[] = [];
    const rpcKey = ["rpc.log", "rpc_log.json", "ipc.log", "rpc.json"].find((k) => files[k] !== undefined);
    if (rpcKey && files[rpcKey]) {
      const rpcLines = files[rpcKey].split("\n").filter((l) => l.trim().length > 0);
      rpcLog = rpcLines.map((line, idx) => {
        try {
          const parsed = JSON.parse(line);
          return {
            id: parsed.id || `rpc-${idx}`,
            direction: parsed.direction || "sent",
            timestamp: parsed.timestamp || new Date().toISOString(),
            method: parsed.method || "unknown",
            params: parsed.params,
            result: parsed.result,
            error: parsed.error,
          };
        } catch (e) {
          return {
            id: `rpc-${idx}`,
            direction: "sent",
            timestamp: new Date().toISOString(),
            method: "raw_log",
            params: { raw: line },
          };
        }
      });
    }

    // Parse terminal.log / stdout.log
    let terminalLog: string[] = [];
    const termKey = ["terminal.log", "stdout.log", "stdout", "stderr.log"].find((k) => files[k] !== undefined);
    if (termKey && files[termKey]) {
      terminalLog = files[termKey].split("\n").filter((l) => l.trim().length > 0);
    }

    // Parse diffs
    let diffs: any[] = [];
    const diffKey = ["diffs.patch", "diff.patch", "patch.diff"].find((k) => files[k] !== undefined);
    if (diffKey && files[diffKey]) {
      diffs = parseDiffText(diffKey, files[diffKey]);
    }

    if (thoughts.length === 0) {
      thoughts = [
        {
          id: "live-fall-1",
          timestamp: new Date().toISOString(),
          text: `Listening to .agent folder. Detected files: ${Object.keys(files).join(", ")}.`,
          phase: status === "idle" ? "perceive" : status,
          category: "system",
        },
      ];
    }

    res.json({
      exists: true,
      path: agentDir,
      session: {
        id: "live-session",
        name: "MachineGPT Live Agent Run",
        timestamp: new Date().toISOString(),
        status,
        elapsedTime: 0,
        stats: {
          promptTokens: 0,
          completionTokens: 0,
          estimatedCost: 0,
          pid,
          hostname: os.hostname(),
          os: process.platform,
        },
        plan,
        thoughts,
        rpcLog,
        terminalLog,
        diffs,
        files,
      },
    });
  } catch (error: any) {
    const message = error?.message || (error ? String(error) : "Failed to load live agent session");
    res.status(500).json({ error: message });
  }
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

  app.listen(PORT, BIND_HOST, () => {
    console.log(`Server running on http://${BIND_HOST}:${PORT}`);
    if (BIND_HOST !== "127.0.0.1" && BIND_HOST !== "localhost" && !API_TOKEN) {
      console.warn(
        "WARNING: dashboard is bound to a non-localhost host without DASHBOARD_API_TOKEN set. " +
        "Anyone reachable on this network can read your .agent folder contents via /api/live-session. " +
        "Set DASHBOARD_API_TOKEN to require authentication."
      );
    }
  });
}

startServer();
