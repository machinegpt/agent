// @vitest-environment node
import { describe, it, expect, beforeEach } from "vitest";
import { timingSafeEqual, createSessionTracker, parseDiffText, _resetSessionTracker } from "../server-utils";

describe("timingSafeEqual", () => {
  it("returns true for equal strings", () => {
    expect(timingSafeEqual("hello", "hello")).toBe(true);
  });

  it("returns false for different strings", () => {
    expect(timingSafeEqual("hello", "world")).toBe(false);
  });

  it("returns false for different lengths", () => {
    expect(timingSafeEqual("abc", "abcd")).toBe(false);
  });

  it("returns true for empty strings", () => {
    expect(timingSafeEqual("", "")).toBe(true);
  });

  it("is case-sensitive", () => {
    expect(timingSafeEqual("Token", "token")).toBe(false);
  });
});

describe("createSessionTracker", () => {
  beforeEach(() => {
    _resetSessionTracker();
  });

  it("returns live-session for first task", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    expect(getOrCreateSessionId("/repo/.agent", "task one")).toBe("live-session");
  });

  it("returns live-session-N for subsequent tasks", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    getOrCreateSessionId("/repo/.agent", "task one");
    expect(getOrCreateSessionId("/repo/.agent", "task two")).toBe("live-session-1");
    expect(getOrCreateSessionId("/repo/.agent", "task three")).toBe("live-session-2");
  });

  it("reuses same ID when task does not change", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    expect(getOrCreateSessionId("/repo/.agent", "same task")).toBe("live-session");
    expect(getOrCreateSessionId("/repo/.agent", "same task")).toBe("live-session");
  });

  it("tracks separate agent directories independently", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    expect(getOrCreateSessionId("/a/.agent", "task A")).toBe("live-session");
    expect(getOrCreateSessionId("/b/.agent", "task B")).toBe("live-session");
    getOrCreateSessionId("/a/.agent", "task A2");
    expect(getOrCreateSessionId("/b/.agent", "task B")).toBe("live-session");
  });

  it("empty task does not trigger a new session", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    getOrCreateSessionId("/repo/.agent", "task one");
    getOrCreateSessionId("/repo/.agent", "");
    expect(getOrCreateSessionId("/repo/.agent", "task one")).toBe("live-session");
  });

  it("transition from empty to first real task keeps same session id", () => {
    const { getOrCreateSessionId } = createSessionTracker();
    getOrCreateSessionId("/repo/.agent", "");
    expect(getOrCreateSessionId("/repo/.agent", "First real task")).toBe("live-session");
    expect(getOrCreateSessionId("/repo/.agent", "First real task")).toBe("live-session");
  });
});

describe("parseDiffText", () => {
  it("returns empty array for empty input", () => {
    const result = parseDiffText("workspace.diff", "");
    expect(result).toEqual([]);
  });

  it("parses a simple unified diff", () => {
    const input = `diff --git a/foo.py b/foo.py
index abc..def 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old line
+new line`;
    const result = parseDiffText("workspace.diff", input);
    expect(result).toHaveLength(1);
    expect(result[0].filepath).toBe("foo.py");
    expect(result[0].filename).toBe("foo.py");
    expect(result[0].additions).toBe(1);
    expect(result[0].deletions).toBe(1);
  });

  it("parses multiple files in a single diff", () => {
    const input = `diff --git a/a.ts b/a.ts
--- a/a.ts
+++ b/a.ts
@@ -1 +1 @@
-old a
+new a
diff --git a/b.ts b/b.ts
--- a/b.ts
+++ b/b.ts
@@ -1 +1,2 @@
 old b
+new line`;
    const result = parseDiffText("workspace.diff", input);
    expect(result).toHaveLength(2);
    expect(result[0].filepath).toBe("a.ts");
    expect(result[0].additions).toBe(1);
    expect(result[0].deletions).toBe(1);
    expect(result[1].filepath).toBe("b.ts");
    expect(result[1].additions).toBe(1);
    expect(result[1].deletions).toBe(0);
  });

  it("handles text without diff headers as a single file", () => {
    const result = parseDiffText("custom.patch", "+added\n-removed\n unchanged");
    expect(result).toHaveLength(1);
    expect(result[0].filepath).toBe("custom.patch");
    expect(result[0].additions).toBe(1);
    expect(result[0].deletions).toBe(1);
  });

  it("counts additions and deletions correctly", () => {
    const input = `diff --git a/code.ts b/code.ts
--- a/code.ts
+++ b/code.ts
@@ -1,5 +1,5 @@
 context
+add1
+add2
-remove1
 context
-remove2
+add3`;
    const result = parseDiffText("workspace.diff", input);
    expect(result).toHaveLength(1);
    expect(result[0].additions).toBe(3);
    expect(result[0].deletions).toBe(2);
  });
});
