/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type SessionStatus =
  | "idle"
  | "perceive"
  | "analyze"
  | "plan"
  | "execute"
  | "verify"
  | "commit"
  | "completed"
  | "error";

export interface PlanStep {
  id: string;
  title: string;
  description: string;
  status: "pending" | "running" | "completed" | "failed";
}

export interface ThoughtLog {
  id: string;
  timestamp: string;
  text: string;
  phase: SessionStatus;
  category: "monologue" | "question" | "decision" | "check" | "system";
}

export interface RPCMessage {
  id: string;
  direction: "sent" | "received";
  timestamp: string;
  method: string;
  params?: any;
  result?: any;
  error?: any;
}

export interface CodeDiff {
  filepath: string;
  filename: string;
  additions: number;
  deletions: number;
  diffText: string;
}

export interface SessionStats {
  promptTokens: number;
  completionTokens: number;
  estimatedCost: number; // in USD
  pid: number;
  hostname: string;
  os: string;
}

export interface AgentSession {
  id: string;
  name: string;
  timestamp: string;
  status: SessionStatus;
  elapsedTime: number; // in seconds
  stats: SessionStats;
  plan: PlanStep[];
  thoughts: ThoughtLog[];
  rpcLog: RPCMessage[];
  terminalLog: string[];
  diffs: CodeDiff[];
  diffsError?: string | null;
  files: Record<string, string>; // raw contents in .agent folder
  summary?: string;
  facts?: string[];
  debt?: string[];
  open?: string[];
  copyCount?: number; // how many times this backup was re-uploaded
}
