import crypto from "crypto";
import { parseDiffText } from "./diff-utils";

export function timingSafeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}

export interface SessionTrackerEntry {
  currentTask: string;
  seq: number;
}

interface SessionTrackerEntryExtended extends SessionTrackerEntry {
  pid: number;
}

const sessionTracker = new Map<string, SessionTrackerEntryExtended>();
let nextPid = 10000 + Math.floor(Math.random() * 9000);

export function createSessionTracker() {
  function getOrCreateSessionId(agentDir: string, task: string): string {
    let tracker = sessionTracker.get(agentDir);
    if (!tracker) {
      tracker = { currentTask: "", seq: 0, pid: nextPid++ };
      sessionTracker.set(agentDir, tracker);
    }

    if (task && tracker.currentTask !== task) {
      // Only increment seq when transitioning from one real task to another.
      // The empty-string → first-real-task transition should NOT change the session ID.
      if (tracker.currentTask !== "") {
        tracker.seq++;
        tracker.pid = nextPid++;
      }
      tracker.currentTask = task;
    }

    return tracker.seq === 0 ? "live-session" : `live-session-${tracker.seq}`;
  }

  function getSessionPid(agentDir: string): number {
    return sessionTracker.get(agentDir)?.pid || 0;
  }

  return { getOrCreateSessionId, getSessionPid };
}

/** Reset session tracker state (for testing only). */
export function _resetSessionTracker() {
  sessionTracker.clear();
  nextPid = 10000 + Math.floor(Math.random() * 9000);
}

export { parseDiffText };
