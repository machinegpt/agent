import crypto from "crypto";

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

export function parseDiffText(filename: string, text: string) {
  if (!text) return [];

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

export type ParseDiffResult = ReturnType<typeof parseDiffText>;
