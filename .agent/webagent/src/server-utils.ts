import crypto from "crypto";
import { parseDiffText } from "./diff-utils";

export function timingSafeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    crypto.timingSafeEqual(bufA, bufA);
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}

export interface SessionTrackerEntry {
  currentTask: string;
  seq: number;
}

const sessionTracker = new Map<string, SessionTrackerEntry>();

export function createSessionTracker() {
  function getOrCreateSessionId(agentDir: string, task: string): string {
    let tracker = sessionTracker.get(agentDir);
    if (!tracker) {
      tracker = { currentTask: "", seq: 0 };
      sessionTracker.set(agentDir, tracker);
    }

    if (task && tracker.currentTask !== task) {
      if (tracker.currentTask !== "") {
        tracker.seq++;
      }
      tracker.currentTask = task;
    }

    return tracker.seq === 0 ? "live-session" : `live-session-${tracker.seq}`;
  }

  return { getOrCreateSessionId };
}

/** Reset session tracker state (for testing only). */
export function _resetSessionTracker() {
  sessionTracker.clear();
}

export { parseDiffText };

export type ParseDiffResult = ReturnType<typeof parseDiffText>;
