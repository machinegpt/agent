<p align="center">
  <img src="https://img.shields.io/badge/machineGPT-Agent_System-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=" />
  <img src="https://img.shields.io/badge/version-2.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-production-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">Agent</h1>
<p align="center"><strong>Autonomous two-agent AI architecture for software projects.<br>Drop one folder. Write one message. Never manage AI context again.</strong></p>

---

## The Problem

Every developer using AI coding assistants hits the same wall:

**The AI forgets everything between sessions.** You start a new conversation and the AI has no idea what language your project uses, what framework you picked, what naming conventions you follow, or why you made that architectural decision three sessions ago. You spend the first 10-15 messages re-explaining context that it already "learned" before.

**The AI contradicts itself.** It suggests patterns that violate conventions it followed yesterday. It proposes changes that undo decisions you already made. Without persistent memory, every session starts from zero.

**The AI can't hold the full picture.** Longer projects become unmanageable. The AI sees the file you're working on but doesn't understand how it connects to the rest of the system. It makes changes that break things three modules away.

**You become the AI's memory.** You end up managing context instead of writing code. Pasting snippets, re-explaining architecture, correcting the same mistakes — every single time.

**machineGPT eliminates all of this.** The `.agent/` folder is a persistent brain that lives inside your project. It remembers everything across every session, forever. You never manage AI context again.

---

## How It Works — The Big Picture

The system has three layers that work together:

### Layer 1: RULES.md — The Unbreakable Rules

This file is read first, every session, no exceptions. It contains 20 absolute prohibitions that no AI model can bypass:

- The AI must always read its memory before coding
- The AI must never write code without a plan
- The AI must always investigate before implementing
- The AI must always validate after changes
- The AI must never exceed the scope of what was planned
- The AI must never add features beyond what was requested
- ...and 15 more

These aren't suggestions. They're enforced constraints. Even if you tell the AI to "just do it" or "skip the planning," it will refuse and explain why. This ensures the system never cuts corners, never takes shortcuts, and never produces unreliable output.

The file also contains a **gate protocol** — a checklist the AI runs before writing any code:

```
□ ACTION file exists for this task?
□ MEMORY conventions known?
□ Dependencies grepped?
□ Scope checked?
□ Validation command known?
```

If any answer is "no," the AI stops and fixes the prerequisite before continuing. This prevents the most common failure mode: writing code before understanding the problem.

### Layer 2: JINX and MACHINE — The Two Agents

The system splits AI work into two specialized roles:

**JINX is the architect.** It thinks. Before writing any code, JINX:

1. **Reframes the task** — figures out what you actually need (which is often different from what you asked for)
2. **Reads the dependency graph** — understands what imports what, what calls what, what will break if changed
3. **Checks architectural invariants** — identifies rules that cannot be violated (e.g., "database access only through the repository layer")
4. **Generates 2-3 approaches** — with explicit trade-offs for each
5. **Picks one** — with a clear reason and what it's sacrificing
6. **Writes a plan** — `PLAN.md` with concrete steps, each mapped to an `ACTION_*.md` file

JINX never writes code. Its job is to make sure the plan is right before anything is built.

**MACHINE is the builder.** It executes. For each ACTION file, MACHINE:

1. **Investigates** — greps for all dependents, traces data flow, identifies the exact divergence point
2. **Implements** — follows conventions exactly, handles edge cases explicitly, one change at a time
3. **Validates** — runs project-specific tests, writes tests that break its own code
4. **Records** — updates memory with new decisions, conventions, debt, failure patterns
5. **Cleans up** — deletes the ACTION file, marks the step complete

MACHINE never redesigns. It executes the plan exactly as specified.

### Layer 3: Persistent Memory — MEMORY.md

This is the shared brain. Every session, both agents read it first. It stores:

- **Project info** — language, framework, architecture, entry points
- **Conventions** — naming patterns, error handling, testing approach
- **Active decisions** — every architectural choice, why it was made, what was rejected
- **Technical debt** — every shortcut, with severity (P0/P1/P2)
- **Failure patterns** — what went wrong, what works, how often it happens
- **Dependency graph** — what imports what (updated by MACHINE during execution)
- **Known constraints** — hard limits that shape every implementation

The memory is never a log. It's never a history. It's always the **current state** of the project. Stale content is deleted immediately. When it gets too large, it compresses — hot data stays, warm data shrinks, cold data archives.

---

## The Feedback Loop — How the System Gets Smarter

The most powerful feature: the system learns from its mistakes across sessions.

**Step 1:** MACHINE executes a task and discovers a dependency that JINX didn't know about.

**Step 2:** MACHINE updates the dependency graph in MEMORY.md and notifies JINX.

**Step 3:** Next time JINX plans a similar task, it reads the updated dependency graph and makes a better plan.

**Step 4:** If the same failure pattern happens 3 times, the system stops and fundamentally reconsiders its approach instead of retrying.

This means the system improves through use — not because the AI model gets smarter, but because the context gets better.

---

## File Structure

```
your-project/
├── .agent/
│   ├── RULES.md        ← Absolute constraints (read first)
│   ├── JINX.md         ← Planning protocols + dependency graph + invariants
│   ├── MACHINE.md      ← Execution protocols + validation + failure modes
│   ├── MEMORY.md       ← Persistent project state
│   ├── PLAN.md         ← Current task checklist
│   ├── PROTOTYPE.md    ← Bootstrap (deleted after first run)
│   └── ACTION_*.md     ← Atomic work units (dynamic, created/deleted)
├── src/
├── package.json
└── ...
```

### What Each File Does

**RULES.md** — The constitution. 20 prohibitions, gate protocol, 19 user action types with fixed responses. Read first. Overrides everything. The AI cannot deviate from these rules.

**JINX.md** — The architect's manual. Contains: boot sequence, reframing protocol, PLAN and ACTION contracts, dependency graph (what imports what), architectural invariant checker, failure intelligence, context compression rules.

**MACHINE.md** — The builder's manual. Contains: boot sequence, execution protocol (investigate → implement → validate → record → cleanup), project validation commands (injected by PROTOTYPE.md), common failure modes (injected by PROTOTYPE.md), code standards, failure recovery classification.

**MEMORY.md** — The shared brain. Read every session. Updated after every meaningful change. Never preserves history — only current state. Critical section (constraints, validation, failure patterns) at the top for fast retrieval.

**PLAN.md** — The current task. Written by JINX. Contains: goal, context, approach, steps with priority and dependencies, blockers, unplanned decisions, outcome. Never deleted — becomes changelog.

**PROTOTYPE.md** — The bootstrap. Only read once — on the very first session. Analyzes your project deeply: maps the dependency graph, finds architectural invariants, discovers validation commands, identifies common failure modes. Injects all of this into JINX.md and MACHINE.md. Then deletes itself.

**ACTION_*.md** — Atomic work units. Created by JINX, executed by MACHINE, deleted after completion. Each contains: exact task, context, dependencies, done-when condition, rollback plan, partial completion notes. If a session crashes mid-task, ACTION files survive and the next session resumes exactly where it stopped.

---

## Installation

### Step 1: Copy the folder

```bash
cp -r .agent /your/project/root/
```

After this, your project looks like:

```
your-project/
├── .agent/
│   ├── RULES.md
│   ├── JINX.md
│   ├── MACHINE.md
│   ├── MEMORY.md
│   ├── PLAN.md
│   ├── PROTOTYPE.md
│   └── (no ACTION files yet)
├── src/
└── ...
```

### Step 2: Initialize (one message)

Open your AI assistant (Claude, Cursor, Windsurf, Aider, or any CLI agent) and send **exactly this once**:

```
Read all files in .agent/. You are now operating as the machineGPT agent system.
Scan this project. Write everything into .agent/MEMORY.md. Confirm when done.
```

What happens:
1. The AI reads all `.agent/` files
2. It scans your project — finds the language, framework, dependencies, conventions
3. It maps the dependency graph (what imports what)
4. It finds architectural invariant rules
5. It discovers validation commands (build, test, lint)
6. It identifies common failure modes for your stack
7. It writes everything into `MEMORY.md`
8. It injects project-specific data into `JINX.md` and `MACHINE.md`
9. It deletes `PROTOTYPE.md`
10. It confirms readiness

**This is the only setup message you'll ever send.**

### Step 3: Use it

From now on, just describe what you want:

```
You:    "Add rate limiting to the API"
Agents: JINX plans → MACHINE implements → done

You:    "There's a bug in the payment flow"
Agents: Diagnose → fix → regression test → done

You:    "How does the auth module work?"
Agents: Read code → answer directly (no plan needed for questions)
```

---

## Day-to-Day Usage

### Normal tasks

```
You:    "Add user authentication with JWT"
Agents: JINX reframes → evaluates approaches → writes plan + ACTION files
        MACHINE investigates → implements → tests → records → done
        MEMORY.md updated with new conventions and decisions
```

### Bug fixes

```
You:    "Orders are duplicating in the payment flow"
Agents: JINX reframes (the real problem might be in the state machine, not the payment code)
        MACHINE traces data flow → finds root cause → fixes → writes regression test
        Failure pattern logged to MEMORY.md for future avoidance
```

### Architecture questions

```
You:    "Should we switch from REST to GraphQL?"
Agents: JINX evaluates → lists trade-offs → makes recommendation
        (No plan is written unless you decide to implement)
```

### Refactoring

```
You:    "The auth module is too large, split it up"
Agents: JINX reads dependency graph → calculates blast radius
        Plans split that respects architectural invariants
        MACHINE executes step by step, validating after each change
```

### What you never need to do

- Never re-explain your stack or conventions
- Never paste code snippets for context
- Never manage AI state between sessions
- Never write system prompts
- Never correct the same mistake twice (failure patterns prevent repetition)

---

## How Memory Works

### Reading

Every session starts by reading `MEMORY.md`. The critical section (constraints, validation commands, failure patterns) is at the top — it's read first and takes priority.

### Writing

After every meaningful change, the relevant section is updated:
- New convention → `## Conventions`
- New decision → `## Decisions`
- New debt → `## Debt`
- New failure pattern → `## Failure Patterns`
- New dependency → `## Dependencies`

Stale content is deleted. History is never preserved.

### Compression

When `MEMORY.md` exceeds 400 lines or hasn't been compressed in 3+ sessions:
1. Full snapshot saved to `ARCHIVE_*.md`
2. Hot data (recent decisions, active work) stays in MEMORY.md
3. Warm data (stable references) compresses to minimal notation
4. Cold data (historical) stays in the archive

If the project is genuinely complex and needs >400 lines of active state, MEMORY.md splits into two files: `MEMORY.md` (hot) + `MEMORY_CONTEXT.md` (warm).

### Dependency Graph

The dependency graph is a map of what imports/calls what. It's:
- **Created** by PROTOTYPE.md during bootstrap
- **Updated** by MACHINE when it discovers new dependencies during execution
- **Read** by JINX when planning — to calculate blast radius

This means JINX knows exactly what will break before proposing a change.

### Architectural Invariants

These are rules that CANNOT be violated. They're discovered by PROTOTYPE.md during bootstrap by analyzing repeated patterns in the source code. Examples:
- "All API routes go through middleware"
- "Database access only through the repository layer"
- "No direct external calls from business logic"

JINX checks every plan against these invariants. If a plan would violate one, it's rejected and replanned.

---

## How Failure Intelligence Works

When something goes wrong — wrong plan, wrong approach, wrong assumption:

1. The failure is logged to `MEMORY.md` → `## Failure Patterns`
2. The pattern includes: trigger, error, fix, frequency count
3. Before planning similar work, JINX checks the pattern library
4. If the same pattern triggers 3 times, the system **stops** and fundamentally reconsiders

This prevents the most common AI failure mode: retrying the same broken approach indefinitely.

---

## How the Session Lifecycle Works

Every session — regardless of what happened before — follows this exact sequence:

```
START
  ↓
Read RULES.md → MEMORY.md → PLAN.md → ACTION_*.md
  ↓
INTEGRITY CHECK: Is MEMORY.md complete? Not truncated?
  (If corrupt → restore from archive or flag for rebuild)
  ↓
STALENESS CHECK: Last updated >3 sessions ago?
  (If stale → compress hot→cold)
  ↓
ORPHAN CHECK: PLAN steps without matching ACTION files?
  (If orphaned → reconcile: recreate or mark done)
  ↓
SYMBOL CHECK: ACTION files reference deleted symbols?
  (If deleted → quarantine the ACTION file)
  ↓
VERSION CHECK: File versions match?
  (If mismatch → log, proceed with newer)
  ↓
MERGE: Completed actions → MEMORY.md → delete ACTION files
  ↓
READY — full project context restored
```

Then for each task:
```
USER: "Add authentication"
  ↓
JINX: Read dependency graph + invariants → reframe → plan → PLAN.md + ACTIONs
  ↓
MACHINE: For each ACTION:
  Investigate → Implement → Validate → Record → Cleanup
  If new dependency found → update JINX dependency graph
  ↓
MEMORY.md updated. Failure patterns logged if any.
```

---

## Compatibility

The system works with any AI assistant that can read files from your project directory:

| Tool | How to use |
|------|-----------|
| **Claude** (claude.ai, API) | Works directly. Files in project root. |
| **Cursor** | Add `.agent` to context with `@.agent` in first message. |
| **Windsurf** | Works directly. Boot reads files automatically. |
| **GitHub Copilot** | Works in Workspace mode. |
| **Aider** | Works with file access. |
| **Any CLI agent** | As long as it can read files, it works. |

The system is **model-agnostic**. It works with Claude, GPT-4, Gemini, Llama, or any capable model. The protocols are written in plain language — any model that can follow instructions can operate under them.

---

## FAQ

**Do I need to mention `.agent/` in every message?**
No. Once initialized, the agents read the folder at the start of every session automatically. You only need to describe your task.

**What if the AI ignores the `.agent/` folder?**
Some tools require you to include the folder in context explicitly (e.g., `@.agent` in Cursor). After the first session, the boot sequence handles this automatically.

**Can multiple developers use this on the same project?**
Yes. Commit `.agent/` to your repository. Every developer's AI sessions will share the same project memory, conventions, and active decisions. The system handles conflicts through a priority protocol.

**What happens if a session ends mid-task?**
The `ACTION_*.md` files survive the session with partial-completion notes. The next session reads them, sees exactly what was done vs what remains, and continues from the failure point. Nothing is lost.

**Can I edit `MEMORY.md` manually?**
Yes, and you should when you want to enforce something permanently — a convention, a decision, a preference. The agents treat it as authoritative.

**What if JINX's plan is wrong?**
Tell the AI the plan is wrong. JINX will reframe and replan. MACHINE will not execute a plan JINX hasn't signed off on.

**What happens when `MEMORY.md` gets too large?**
The compression protocol activates. Hot data stays. Warm data compresses to minimal notation. Cold data archives to separate files. If the project is genuinely complex, MEMORY.md splits into two files. Context is never lost — it's compressed intelligently.

**How does the system learn from mistakes?**
When a failure occurs, it's logged as a failure pattern. Before planning similar work, the agents check the pattern library. If the same pattern triggers 3 times, the system stops and reconsiders the fundamental approach instead of retrying.

**What if the agent system itself needs upgrading?**
Version fields in each file track the template version. On boot, version mismatches are logged. New sections get placeholders. Old sections are preserved until compression. Migration history is tracked in MEMORY.md.

**What does PROTOTYPE.md actually analyze?**
During bootstrap, it: maps the dependency graph (what imports what), finds architectural invariants (what can't change), discovers validation commands (how to build/test/lint), identifies common failure modes (what frequently breaks in this stack). All of this is injected into JINX.md and MACHINE.md so they work with your specific project.

**What if my project is empty (no source files yet)?**
PROTOTYPE.md handles this gracefully. It notes "new project, no source" and leaves relevant fields as `—`. The system is ready to go when you add code.

**Can I use this with a monorepo?**
Yes. The dependency graph and directory index sections scale to multiple packages. PROTOTYPE.md maps the full graph during bootstrap.

---

## Design Philosophy

**AI-assisted development fails not because models aren't capable, but because they have no memory and no structure.**

A capable model working from a fresh context window will produce plausible code that contradicts your architecture, ignores your conventions, and forgets the decision you made three sessions ago. It cannot help it — it has no state.

The `.agent/` system gives the model state. Not by making the model smarter, but by making the context persistent, structured, and authoritative. The agents don't need to be reminded what language you use. They don't need to be told your naming conventions again. They read `MEMORY.md` and they know.

**The RULES.md layer** ensures the system cannot be bypassed. No model can skip boot, ignore conventions, or deviate from protocols. The constraints are absolute.

**The JINX/MACHINE split** forces planning before execution. Thinking and doing are different cognitive modes. Separating them eliminates the architecturally wrong implementation that was technically correct.

**The dependency graph** gives JINX awareness of the actual codebase structure. Plans respect real dependencies, not assumptions.

**The failure intelligence** means the system improves across sessions — not because the model improves, but because the context improves.

**The context compression** ensures the system never overflows. Signal density stays high regardless of project age or complexity.

---

<p align="center">
  <strong>machineGPT</strong> · Agent System · v2.0.0<br>
  <em>Built for engineers who want AI that works like a senior teammate, not a stateless autocomplete.</em>
</p>
