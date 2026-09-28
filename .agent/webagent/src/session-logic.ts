/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentSession, SessionStatus } from "./types";

/** Statuses that mean a run has finished and should be archived out of the live slot. */
export function isTerminalStatus(status: SessionStatus): boolean {
  return status === "completed" || status === "error";
}

/**
 * Stable identity for "this exact run reached this exact terminal state".
 *
 * The live slot id is recycled by the server (`live-session`, `live-session-1`,
 * … one per task), so the id alone cannot distinguish a finished run from a
 * fresh one that happens to reuse it. Pairing it with the terminal status does:
 * the pair is derived from data that is persisted with the archive, which is
 * what makes duplicate detection survive a page reload.
 */
export function terminalKeyOf(session: AgentSession): string {
  return `${session.id}:${session.status}`;
}

/** True when some entry in the list is already the archive of this terminal run. */
export function hasArchivedTerminal(sessions: AgentSession[], terminalKey: string): boolean {
  return sessions.some((s) => s.terminalKey === terminalKey);
}

/**
 * Merges a live payload into the session list.
 *
 * An existing entry with the same id is replaced in place; otherwise the payload
 * takes the live slot and every idle live- placeholder is dropped, so a restart
 * of JINX never leaves two idle monitors behind.
 */
export function mergeLiveSession(prev: AgentSession[], newSession: AgentSession): AgentSession[] {
  const existingIdx = prev.findIndex((s) => s.id === newSession.id);
  if (existingIdx >= 0) {
    const merged = [...prev];
    merged[existingIdx] = newSession;
    return merged;
  }
  return [
    newSession,
    ...prev.filter((s) => !(s.id.startsWith("live-") && s.status === "idle")),
  ];
}

/**
 * Archives a finished run and hands the live slot back as an idle placeholder.
 *
 * Returns `prev` untouched when this terminal run is already archived, which
 * makes the caller safe to invoke on every poll: two payloads arriving in the
 * same tick both see the same authoritative list inside the state updater, so
 * the second one is a no-op rather than a second copy of the same run.
 */
export function archiveTerminalSession(
  prev: AgentSession[],
  session: AgentSession,
  now: number,
  makeIdleLive: () => AgentSession
): AgentSession[] {
  const terminalKey = terminalKeyOf(session);
  if (hasArchivedTerminal(prev, terminalKey)) {
    return prev;
  }
  const archived: AgentSession = {
    ...session,
    id: `${session.status}-${now}`,
    copyCount: 0,
    terminalKey,
  };
  return [makeIdleLive(), archived, ...prev.filter((s) => s.id !== session.id)];
}

/**
 * Enforces the invariant the history list depends on: a live- slot never holds a
 * finished run.
 *
 * A terminal run lives in exactly one place — the archive taken when it
 * finished. Once that archive exists, a live- entry carrying the same terminal
 * status is a second copy of the same run and is dropped. This also cleans up
 * duplicates written by earlier builds, which had no terminalKey to compare on;
 * nothing is lost because the archive is a full copy of the session.
 */
export function dropDuplicateTerminalRuns(sessions: AgentSession[]): AgentSession[] {
  const archivedStatuses = new Set(
    sessions.filter((s) => !s.id.startsWith("live-") && isTerminalStatus(s.status)).map((s) => s.status)
  );
  if (archivedStatuses.size === 0) return sessions;

  const seenTerminalKeys = new Set<string>();
  return sessions.filter((s) => {
    if (s.terminalKey) {
      if (seenTerminalKeys.has(s.terminalKey)) return false;
      seenTerminalKeys.add(s.terminalKey);
      return true;
    }
    if (s.id.startsWith("live-") && isTerminalStatus(s.status) && archivedStatuses.has(s.status)) {
      return false;
    }
    return true;
  });
}
