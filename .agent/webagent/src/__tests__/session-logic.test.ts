/**
 * Regression tests for the duplicate entries that appeared in "История сессий"
 * while JINX was running.
 *
 * The scenario: the dashboard server keeps serving a finished run's payload on
 * every poll, because the .agent files stay on disk after the run ends. The
 * dashboard archived that run once, but then kept re-merging the same payload
 * into the live slot, so the history list ended up holding the same run twice —
 * once as the archive, once as a live- entry with terminal status.
 */
import { describe, it, expect } from "vitest";
import { AgentSession, SessionStatus } from "../types";
import {
  isTerminalStatus,
  terminalKeyOf,
  hasArchivedTerminal,
  mergeLiveSession,
  archiveTerminalSession,
  dropDuplicateTerminalRuns,
} from "../session-logic";

function makeSession(overrides: Partial<AgentSession> = {}): AgentSession {
  return {
    id: "live-session",
    name: "MachineGPT Live Agent Run",
    timestamp: "2026-09-28T13:00:00.000Z",
    status: "execute",
    elapsedTime: 10,
    stats: {
      promptTokens: 0,
      completionTokens: 0,
      estimatedCost: 0,
      pid: 1234,
      hostname: "localhost",
      os: "local",
    },
    plan: [],
    thoughts: [],
    rpcLog: [],
    terminalLog: [],
    diffs: [],
    files: {},
    ...overrides,
  };
}

const makeIdleLive = () => makeSession({ id: "live-session", status: "idle", timestamp: "2026-09-28T14:00:00.000Z" });

describe("isTerminalStatus", () => {
  it("treats completed and error as terminal", () => {
    expect(isTerminalStatus("completed")).toBe(true);
    expect(isTerminalStatus("error")).toBe(true);
  });

  it("treats every other status as running", () => {
    for (const s of ["idle", "perceive", "analyze", "plan", "execute", "verify", "commit"] as const) {
      expect(isTerminalStatus(s)).toBe(false);
    }
  });
});

describe("terminalKeyOf", () => {
  it("pairs the recycled live id with the status so runs stay distinguishable", () => {
    expect(terminalKeyOf(makeSession({ id: "live-session", status: "completed" }))).toBe("live-session:completed");
    expect(terminalKeyOf(makeSession({ id: "live-session-1", status: "completed" }))).toBe("live-session-1:completed");
  });
});

describe("mergeLiveSession", () => {
  it("replaces an existing entry with the same id in place", () => {
    const prev = [makeSession({ status: "execute" })];
    const next = makeSession({ status: "completed" });
    const result = mergeLiveSession(prev, next);
    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("completed");
  });

  it("drops idle live- placeholders when a new live id arrives", () => {
    const prev = [makeSession({ id: "live-session", status: "idle" })];
    const result = mergeLiveSession(prev, makeSession({ id: "live-session-1" }));
    expect(result.map((s) => s.id)).toEqual(["live-session-1"]);
  });

  it("keeps a finished live entry when a new run takes the slot", () => {
    const prev = [makeSession({ id: "live-session", status: "completed" })];
    const result = mergeLiveSession(prev, makeSession({ id: "live-session-1" }));
    expect(result.map((s) => s.id)).toEqual(["live-session-1", "live-session"]);
  });
});

describe("archiveTerminalSession", () => {
  it("archives the run, stamps a terminalKey, and hands back an idle live slot", () => {
    const prev = [makeSession({ status: "completed" })];
    const result = archiveTerminalSession(prev, makeSession({ status: "completed" }), 1750000000000, makeIdleLive);

    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("live-session");
    expect(result[0].status).toBe("idle");
    expect(result[1].id).toBe("completed-1750000000000");
    expect(result[1].terminalKey).toBe("live-session:completed");
    expect(result[1].copyCount).toBe(0);
  });

  it("is a no-op when the same terminal run is already archived", () => {
    const first = archiveTerminalSession([], makeSession({ status: "completed" }), 1750000000000, makeIdleLive);
    const second = archiveTerminalSession(first, makeSession({ status: "completed" }), 1750000009999, makeIdleLive);

    expect(second).toBe(first);
    expect(second.filter((s) => s.terminalKey === "live-session:completed")).toHaveLength(1);
  });

  it("keeps a full copy of the run in the archive", () => {
    const run = makeSession({ status: "completed", terminalLog: ["[JINX_COMPLETE]"], thoughts: [{ id: "t1", timestamp: "x", text: "done", phase: "commit", category: "system" }] });
    const result = archiveTerminalSession([], run, 1, makeIdleLive);
    const archive = result.find((s) => s.terminalKey)!;
    expect(archive.terminalLog).toEqual(["[JINX_COMPLETE]"]);
    expect(archive.thoughts).toHaveLength(1);
  });
});

describe("hasArchivedTerminal", () => {
  it("finds the archive by terminalKey and ignores unrelated entries", () => {
    const list = archiveTerminalSession([], makeSession({ status: "completed" }), 1, makeIdleLive);
    expect(hasArchivedTerminal(list, "live-session:completed")).toBe(true);
    expect(hasArchivedTerminal(list, "live-session:error")).toBe(false);
    expect(hasArchivedTerminal(list, "live-session-1:completed")).toBe(false);
  });
});

describe("the duplicate the dashboard actually showed", () => {
  it("does not re-merge a terminal payload after it has been archived", () => {
    // Round 1: the run finishes and gets archived.
    let list = archiveTerminalSession([makeSession()], makeSession({ status: "completed" }), 1750000000000, makeIdleLive);
    expect(list.map((s) => s.id)).toEqual(["live-session", "completed-1750000000000"]);

    // Round 2..N: the server keeps replaying the same finished payload. The
    // guard is the fix — without it the live slot is refilled and the run
    // appears twice.
    for (let i = 0; i < 5; i++) {
      const replayed = makeSession({ status: "completed" });
      const key = terminalKeyOf(replayed);
      list = isTerminalStatus(replayed.status) && hasArchivedTerminal(list, key)
        ? list
        : mergeLiveSession(list, replayed);
    }

    expect(list.map((s) => s.id)).toEqual(["live-session", "completed-1750000000000"]);
    expect(list[0].status).toBe("idle");
  });

  it("still shows a terminal run that has not been archived yet", () => {
    // A cold start where JINX already finished: surface the run, but only once.
    // The live slot holds it (same id, replaced in place) until the next poll
    // sees a status transition and archives it.
    const terminal = makeSession({ status: "completed" });
    let list = [makeIdleLive()];
    if (!(isTerminalStatus(terminal.status) && hasArchivedTerminal(list, terminalKeyOf(terminal)))) {
      list = mergeLiveSession(list, terminal);
    }
    expect(list).toHaveLength(1);
    expect(list[0].id).toBe("live-session");
    expect(list[0].status).toBe("completed");
  });

  it("survives a page reload, because the terminalKey is persisted", () => {
    const before = archiveTerminalSession([], makeSession({ status: "completed" }), 1750000000000, makeIdleLive);
    const reloaded: AgentSession[] = JSON.parse(JSON.stringify(before)); // localStorage round-trip

    const replayed = makeSession({ status: "completed" });
    const guarded = isTerminalStatus(replayed.status) && hasArchivedTerminal(reloaded, terminalKeyOf(replayed));
    expect(guarded).toBe(true);
    expect(reloaded.filter((s) => s.terminalKey === "live-session:completed")).toHaveLength(1);
  });

  it("keeps consecutive runs separate instead of collapsing them", () => {
    let list = archiveTerminalSession([makeSession()], makeSession({ status: "completed" }), 1, makeIdleLive);
    // The next task recycles a new live id; its own terminal run must archive too.
    list = mergeLiveSession(list, makeSession({ id: "live-session-1" }));
    list = archiveTerminalSession(list, makeSession({ id: "live-session-1", status: "completed" }), 2, makeIdleLive);

    expect(list.map((s) => s.id)).toEqual(["live-session", "completed-2", "completed-1"]);
  });
});

describe("dropDuplicateTerminalRuns", () => {
  it("removes the legacy live- copy of a run that is already archived", () => {
    // The shape written by the previous build: no terminalKey anywhere, and the
    // live slot holding a finished run next to its archive.
    const legacy = [
      makeSession({ id: "live-session", status: "completed" }),
      makeSession({ id: "live-session-1", status: "completed" }),
      makeSession({ id: "completed-1750000000000", status: "completed" }),
      makeSession({ id: "completed-1750000009999", status: "completed" }),
    ];

    const result = dropDuplicateTerminalRuns(legacy);
    expect(result.map((s) => s.id)).toEqual(["completed-1750000000000", "completed-1750000009999"]);
  });

  it("keeps a terminal live run that has no archive yet", () => {
    const list = [makeSession({ id: "live-session", status: "completed" })];
    expect(dropDuplicateTerminalRuns(list)).toHaveLength(1);
  });

  it("does not touch lists that are already correct", () => {
    const list = archiveTerminalSession([], makeSession({ status: "completed" }), 1, makeIdleLive);
    expect(dropDuplicateTerminalRuns(list)).toEqual(list);
  });

  it("keeps a running live slot alongside finished archives", () => {
    const list = [
      makeSession({ id: "live-session-1", status: "execute" }),
      makeSession({ id: "completed-1", status: "completed" }),
    ];    expect(dropDuplicateTerminalRuns(list)).toHaveLength(2);
  });

  it("collapses repeated terminalKeys, keeping the first", () => {
    const archived = archiveTerminalSession([], makeSession({ status: "completed" }), 1, makeIdleLive);
    const clone: AgentSession = { ...archived[1], id: "completed-2" };
    const result = dropDuplicateTerminalRuns([...archived, clone]);
    expect(result.filter((s) => s.terminalKey === "live-session:completed")).toHaveLength(1);
  });

  it("is self-healing: a run it drops for lack of an archive comes back on the next poll", () => {
    // The cleanup rule is deliberately conservative - it cannot tell which
    // terminal live entry is stale legacy data and which is a run whose archive
    // has not been written yet. It must therefore never lose a run for good.
    const list = [
      makeSession({ id: "live-session-1", status: "completed" }),
      makeSession({ id: "completed-1", status: "completed" }),
    ];
    const cleaned = dropDuplicateTerminalRuns(list);
    expect(cleaned.some((s) => s.id === "live-session-1")).toBe(false);

    const replayed = makeSession({ id: "live-session-1", status: "completed" });
    const key = terminalKeyOf(replayed);
    const afterPoll = isTerminalStatus(replayed.status) && hasArchivedTerminal(cleaned, key)
      ? cleaned
      : mergeLiveSession(cleaned, replayed);
    expect(afterPoll.some((s) => s.id === "live-session-1")).toBe(true);
  });
});

/**
 * Drives the poll sequence the dashboard actually receives across several JINX
 * runs and reloads, using the same branch structure as applyLiveSessionData, and
 * asserts the invariant the history list depends on: no run is ever listed
 * twice, and a live- slot never holds a finished run that already has an archive.
 */
describe("history list invariant across consecutive runs and reloads", () => {
  function step(
    state: { sessions: AgentSession[]; prevStatus: SessionStatus | null },
    payload: AgentSession
  ): { sessions: AgentSession[]; prevStatus: SessionStatus | null } {
    const isTerminal = isTerminalStatus(payload.status);
    const key = terminalKeyOf(payload);
    const alreadyArchived = (list: AgentSession[]) => isTerminal && hasArchivedTerminal(list, key);

    if (state.prevStatus === null) {
      const next = alreadyArchived(state.sessions) ? state.sessions : mergeLiveSession(state.sessions, payload);
      return { sessions: next, prevStatus: payload.status };
    }
    if (isTerminal && state.prevStatus !== payload.status) {
      return {
        sessions: archiveTerminalSession(state.sessions, payload, 1000 + state.sessions.length, makeIdleLive),
        prevStatus: payload.status,
      };
    }
    return {
      sessions: alreadyArchived(state.sessions) ? state.sessions : mergeLiveSession(state.sessions, payload),
      prevStatus: payload.status,
    };
  }

  /** A page reload: the list comes back through localStorage, refs are gone. */
  function reload(state: { sessions: AgentSession[]; prevStatus: SessionStatus | null }) {
    return {
      sessions: JSON.parse(JSON.stringify(state.sessions)),
      prevStatus: null,
    };
  }

  function assertNoDuplicates(sessions: AgentSession[]) {
    const archivedKeys = sessions.filter((s) => s.terminalKey).map((s) => s.terminalKey!);
    expect(new Set(archivedKeys).size).toBe(archivedKeys.length);

    const archivedStatuses = new Set(
      sessions.filter((s) => !s.id.startsWith("live-") && isTerminalStatus(s.status)).map((s) => s.status)
    );
    for (const s of sessions) {
      if (s.id.startsWith("live-") && isTerminalStatus(s.status)) {
        expect(archivedStatuses.has(s.status)).toBe(false);
      }
    }
  }

  it("holds across two runs, endless replays, and a reload between them", () => {
    let state: { sessions: AgentSession[]; prevStatus: SessionStatus | null } = {
      sessions: [makeIdleLive()],
      prevStatus: null,
    };

    // Run 1: live-session starts, works, finishes, then the server replays it.
    for (const status of ["execute", "verify", "completed"] as const) {
      state = step(state, makeSession({ id: "live-session", status }));
      assertNoDuplicates(state.sessions);
    }
    for (let i = 0; i < 20; i++) {
      state = step(state, makeSession({ id: "live-session", status: "completed" }));
      assertNoDuplicates(state.sessions);
    }
    expect(state.sessions.filter((s) => s.id === "live-session")).toHaveLength(1);
    expect(state.sessions.filter((s) => s.terminalKey === "live-session:completed")).toHaveLength(1);

    // Reload: the archive comes back from localStorage, the in-memory guards do not.
    state = reload(state);
    assertNoDuplicates(state.sessions);
    for (let i = 0; i < 10; i++) {
      state = step(state, makeSession({ id: "live-session", status: "completed" }));
      assertNoDuplicates(state.sessions);
    }
    expect(state.sessions).toHaveLength(2);

    // Run 2: a new task gets a new live id and fails instead of completing.
    for (const status of ["execute", "error"] as const) {
      state = step(state, makeSession({ id: "live-session-1", status }));
      assertNoDuplicates(state.sessions);
    }
    for (let i = 0; i < 20; i++) {
      state = step(state, makeSession({ id: "live-session-1", status: "error" }));
      assertNoDuplicates(state.sessions);
    }
    state = reload(state);
    for (let i = 0; i < 10; i++) {
      state = step(state, makeSession({ id: "live-session-1", status: "error" }));
      assertNoDuplicates(state.sessions);
    }

    // Two distinct runs, two entries - not four.
    expect(state.sessions.map((s) => s.id).sort()).toEqual(["completed-1000", "error-1000"]);
  });
});
