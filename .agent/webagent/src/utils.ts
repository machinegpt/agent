/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import YAML from "yaml";
import { AgentSession, PlanStep, ThoughtLog, RPCMessage, CodeDiff, SessionStatus } from "./types";
import { parseDiffText } from "./diff-utils";

// Load saved sessions from LocalStorage
export function getSavedSessions(): AgentSession[] {
  try {
    const raw = localStorage.getItem("jinx_sessions");
    return raw ? JSON.parse(raw) : [];
  } catch (error) {
    console.error("Failed to load saved sessions from LocalStorage", error);
    return [];
  }
}

// Save sessions list to LocalStorage
export function saveSessions(sessions: AgentSession[]): void {
  try {
    localStorage.setItem("jinx_sessions", JSON.stringify(sessions));
  } catch (error) {
    console.error("Failed to save sessions list to LocalStorage", error);
  }
}

// Default "live monitoring" placeholder session used while waiting
// for the Python JINX agent to publish its first /api/live-session payload.
export function createDefaultLiveSession(): AgentSession {
  return {
    id: "live-session",
    name: "MachineGPT Live Agent Run",
    timestamp: new Date().toISOString(),
    status: "idle",
    elapsedTime: 0,
    stats: {
      promptTokens: 0,
      completionTokens: 0,
      estimatedCost: 0,
      pid: 0,
      hostname: "localhost",
      os: "local",
    },
    plan: [],
    thoughts: [
      {
        id: "system-init",
        timestamp: new Date().toISOString(),
        text: "Listening for local agent activity...",
        phase: "idle",
        category: "system",
      },
    ],
    rpcLog: [],
    terminalLog: [],
    diffs: [],
    files: {},
  };
}

// Parse imported files list from folder selection to create a structured session
export async function parseAgentFolder(filesList: File[]): Promise<AgentSession> {
  const sessionFiles: Record<string, string> = {};
  let status: SessionStatus = "idle";
  let pid = Math.floor(Math.random() * 50000) + 10000;
  let plan: PlanStep[] = [];
  let thoughts: ThoughtLog[] = [];
  let rpcLog: RPCMessage[] = [];
  let terminalLog: string[] = [];
  let diffs: CodeDiff[] = [];

  // Read content of each file
  for (const file of filesList) {
    // webkitRelativePath looks like "some_folder/.agent/plan.json" or "plan.json"
    const relativePath = file.webkitRelativePath || file.name;
    // Extract everything after '.agent/' if present, otherwise use the filename
    const parts = relativePath.split(".agent/");
    const filename = parts.length > 1 ? parts[1] : relativePath.split("/").pop() || relativePath;

    if (!filename || filename.startsWith(".")) continue; // skip hidden files

    const content = await file.text();
    sessionFiles[filename] = content;

    // Parser for JINX standard files
    if (filename === "state.yaml" || filename === "state.json") {
      try {
        const stateObj = YAML.parse(content);
        if (stateObj.phase) status = stateObj.phase.toLowerCase() as SessionStatus;
        if (stateObj.active_pid) pid = stateObj.active_pid;
        if (stateObj.errors && stateObj.errors.length > 0) {
          status = "error";
          terminalLog.push(`[ERROR] ${stateObj.errors.join(", ")}`);
        }
      } catch (e) {
        console.error("Failed to parse state file", e);
      }
    } else if (filename === "plan.yaml" || filename === "plan.json") {
      try {
        const planObj = YAML.parse(content);
        if (Array.isArray(planObj)) {
          plan = planObj.map((p: any, idx: number) => ({
            id: p.id || `step-${idx}`,
            title: p.title || p.name || "Untitled Step",
            description: p.description || p.desc || "",
            status: p.status || "pending",
          }));
        }
      } catch (e) {
        console.error("Failed to parse plan file", e);
      }
    } else if (filename === "thoughts.yaml" || filename === "thought.yaml" || filename === "thoughts.json" || filename === "thought.json") {
      try {
        const thoughtObj = YAML.parse(content);
        if (Array.isArray(thoughtObj)) {
          thoughts = thoughtObj.map((t: any, idx: number) => ({
            id: t.id || `thought-${idx}`,
            timestamp: t.timestamp || new Date().toISOString(),
            text: t.text || t.thought || "",
            phase: (t.phase || "execute").toLowerCase() as SessionStatus,
            category: t.category || "monologue",
          }));
        }
      } catch (e) {
        console.error("Failed to parse thoughts file", e);
      }
    } else if (filename === "rpc.log" || filename === "rpc_log.json" || filename === "ipc.log") {
      try {
        const rpcLines = content.split("\n").filter(l => l.trim().length > 0);
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
            } as RPCMessage;
          }
        });
      } catch (e) {
        console.error("Failed to parse rpc logs", e);
      }
    } else if (filename === "terminal.log" || filename === "stdout.log") {
      terminalLog = content.split("\n").filter(l => l.trim().length > 0);
    } else if (filename === "diffs.patch" || filename === "diff.patch") {
      diffs = parseDiffText(filename, content);
    }
  }

  // If no files matched structured components, populate basic logs
  if (terminalLog.length === 0 && Object.keys(sessionFiles).length > 0) {
    terminalLog.push(`[JINX UI] Synced ${Object.keys(sessionFiles).length} files from local folder.`);
    // generate mock steps
    plan = [
      { id: "s1", title: "Ingest Local Files", description: "Read files from dragged directory", status: "completed" },
    ];
  }

  // Generate fallback thoughts if empty
  if (thoughts.length === 0) {
    thoughts = [
      {
        id: "fall-1",
        timestamp: new Date().toISOString(),
        text: `Loaded local .agent folder snapshot. Active workspace contains files: ${Object.keys(sessionFiles).join(", ")}.`,
        phase: status === "idle" ? "perceive" : status,
        category: "system",
      },
    ];
  }

  const folderName = filesList[0]?.webkitRelativePath?.split("/")[0] || "imported-agent-run";

  return {
    id: `run-import-${Date.now()}`,
    name: `Local Run: ${folderName}`,
    timestamp: new Date().toISOString(),
    status: status === "idle" ? "completed" : status,
    elapsedTime: 45,
    stats: {
      promptTokens: Math.floor(Math.random() * 8000) + 1000,
      completionTokens: Math.floor(Math.random() * 2000) + 300,
      estimatedCost: 0.015,
      pid,
      hostname: "localhost",
      os: "local-browser",
    },
    plan,
    thoughts,
    rpcLog,
    terminalLog,
    diffs,
    files: sessionFiles,
  };
}
