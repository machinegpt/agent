<p align="center">
  <img src="https://img.shields.io/badge/machineGPT-Agent_System-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=" />
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-production-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">Agent</h1>
<p align="center"><strong>Autonomous two-agent AI architecture for software projects.<br>Drop one folder. Write one message. Never manage AI context again.</strong></p>

---

## What this is

machineGPT Agent System is a persistent, self-organizing AI execution layer that lives inside your project. It consists of two specialized agents — **JINX** and **MACHINE** — that collaborate autonomously to plan, implement, test, and remember every decision made in your codebase.

Once installed, the system runs without prompts, without repetition, and without you managing AI state. You describe what you want. The agents handle the rest.

---

## The problem it solves

Every developer working with AI assistants hits the same wall:

- You start a new session and the AI has forgotten everything about your project
- You waste the first 10 messages re-explaining architecture, conventions, and context
- The AI suggests changes that contradict decisions made two sessions ago
- You juggle files, paste code snippets, re-explain stack choices — every single time
- Longer projects become unmanageable because the AI can't hold the full picture

**machineGPT eliminates this entirely.**

The `.agent/` folder is the AI's persistent brain. It stores your project's architecture, conventions, active decisions, and the exact state of every task — across every session, forever.

---

## How it works

The system is built on three principles:

### 1. Persistent shared memory
Every meaningful decision, convention, and architectural choice is written to `MEMORY.md` after it's made. Every session starts by reading it. The AI never forgets your stack, your naming conventions, your error handling patterns, or why you made the choices you made.

### 2. Structured planning before execution
Before writing a single line of code, **JINX** frames the real problem (which is often not the stated one), evaluates approaches, chooses one, and produces a concrete plan. That plan lives in `PLAN.md` as a checklist. Each step maps to an `ACTION_*.md` file — a precise, unambiguous instruction for **MACHINE**.

### 3. Atomic execution with no lost state
**MACHINE** picks up `ACTION` files, executes them fully — with edge cases, tests, and typed code — then marks the step complete and removes the file. If a session ends mid-task, the next session reads the remaining `ACTION` files and continues exactly where it stopped.

---

## The two agents

### JINX — Architecture & Planning
Jinx is the creative and structural half of the system. It does not write code. It thinks.

- Reframes tasks into their real shape before anyone acts on them
- Generates multiple approaches and evaluates trade-offs explicitly
- Writes `PLAN.md` and all `ACTION_*.md` files before execution begins
- Speaks first on every new task; speaks again only when something is wrong structurally
- Flags assumptions that will break the system three weeks from now

JINX's value is not speed. It's the avoidance of expensive mistakes made by acting before understanding.

### MACHINE — Implementation & Execution
MACHINE is the execution half. It does not redesign. It delivers.

- Reads the current `ACTION` file and implements it exactly as specified
- Handles every edge case explicitly — no silent failures, no optimistic assumptions
- Writes the test that breaks its own code before declaring done
- Updates `MEMORY.md` with any conventions or decisions introduced during implementation
- Records technical debt when it creates it; removes it from the log when resolved

MACHINE's value is not creativity. It's reliability: correct, typed, tested, maintainable output every time.

---

## File structure

```
.agent/
├── JINX.md         — JINX identity, protocols, and contracts
├── MACHINE.md      — MACHINE identity, protocols, and contracts
├── MEMORY.md       — Persistent project state (shared, always current)
├── PLAN.md         — Current task goal and step checklist
├── ACTION_*.md     — Atomic work units (dynamic set, created by JINX, deleted by MACHINE)
└── README.md       — This file
```

### What each file does

| File | Created by | Updated by | Deleted |
|------|-----------|-----------|---------|
| `MEMORY.md` | JINX (first session) | Both agents | Never |
| `PLAN.md` | JINX | JINX + MACHINE | Never (becomes changelog) |
| `ACTION_*.md` | JINX | MACHINE (marks done) | MACHINE (after completion) |

---

## Session lifecycle

Every session — regardless of what was done before — follows this sequence:

```
START
  ↓
Both agents read MEMORY.md → PLAN.md → all ACTION_*.md
  ↓
Completed actions merged into MEMORY.md
  ↓
Consumed ACTION files deleted
  ↓
Incomplete plan steps preserved exactly
  ↓
READY — full project context restored
```

Then for each task:

```
USER: "Add user authentication"
  ↓
JINX: Reframes task → evaluates approaches → writes PLAN.md + all ACTION files
  ↓
MACHINE: Reads ACTION files in order → implements → tests → marks step done → deletes file
  ↓
MEMORY.md updated with all new conventions and decisions
```

---

## Why this architecture works

### Separation of concerns at the agent level
Most AI systems use a single model for both thinking and doing. This creates a fundamental tension: the model optimizes for the appearance of progress rather than correctness of direction. By splitting responsibilities — JINX thinks, MACHINE executes — each agent is optimized for exactly one type of output, and neither interferes with the other.

### State that survives session boundaries
The root failure of AI-assisted development is statelessness. Every context window ends and everything is lost. `MEMORY.md` is the solution: a structured, always-current document that both agents read before acting and update after acting. It is not a log. It is not a history. It is the living state of the project — stale content is deleted immediately.

### Plans as contracts, not suggestions
`PLAN.md` is not a scratchpad. It is a formal contract between JINX and MACHINE. Every step maps to exactly one `ACTION` file. Every `ACTION` file has a `Done when` condition with no ambiguity. MACHINE cannot "almost" complete a task. It either satisfies the condition or it does not.

### Debt that can't be ignored
When MACHINE introduces technical debt — a shortcut taken for speed, a TODO left for later — it records it in `MEMORY.md` immediately. It is not possible to forget debt that is written in the first file read every session.

### Zero prompt overhead after setup
Once initialized, you never write system prompts. You never paste context. You never re-explain your stack. The `.agent/` folder contains everything the AI needs to operate with full project awareness from the first token of every session.

---

## Installation
Copy the .agent folder into your project root
```bash
cp -r .agent /your/project/root/
```

Your project structure after installation:

```
your-project/
├── .agent/          ← machineGPT Agent System
│   ├── JINX.md
│   ├── MACHINE.md
│   ├── MEMORY.md
│   ├── PLAN.md
│   └── README.md
├── src/
├── package.json
└── ...
```

---

## First-time setup — one message

After dropping `.agent/` into your project, open your AI assistant (Claude, Cursor, Windsurf, or any CLI agent) and send **exactly this message once**:

```
Read all files in .agent/ — MEMORY.md, PLAN.md, JINX.md, MACHINE.md.
You are now operating as the machineGPT agent system.
Scan this project, infer the language, framework, architecture, and conventions.
Write everything you find into .agent/MEMORY.md.
Confirm when done.
```

The agents will scan your project, populate `MEMORY.md` with everything they find, and confirm readiness. From that point forward — every session, every task — just describe what you want.

---

## Day-to-day usage

After the one-time setup, your workflow is:

```
You:    "Add rate limiting to the API endpoints"
Agents: plan → implement → test → done. MEMORY.md updated.

You:    "Refactor the auth module, it's getting too large"
Agents: Analyze → propose split → implement → test → done.

You:    "There's a bug in the payment flow — orders are duplicating"
Agents: Diagnose → locate → fix → write regression test → done.

You:    "We're switching from REST to GraphQL"
Agents: JINX evaluates migration approach → writes phased plan →
        MACHINE executes step by step → MEMORY.md updated.
```

No context-setting. No re-explaining. No managing AI state. Just tasks.

---

## What gets remembered automatically

`MEMORY.md` is updated continuously throughout your project's life. The agents record:

- **Language and version** — your exact runtime (Node 20, Python 3.12, Go 1.22, etc.)
- **Framework and stack** — every library and framework in use
- **Architecture** — your patterns, layers, and structural decisions
- **Naming conventions** — how files, functions, variables, and routes are named
- **Error handling** — your project's specific approach to errors and exceptions
- **Testing strategy** — where tests live, how they're structured, what framework you use
- **Active decisions** — every architectural choice, what it was, why it was made, who decided
- **Technical debt** — every known shortcut, with context for when it should be resolved
- **User preferences** — behavioral preferences you've stated explicitly

Nothing is inferred and discarded. Everything meaningful is written and kept.

---

## Compatibility

machineGPT Agent System works with any AI assistant that can read files from your project directory:

| Tool | Compatible |
|------|-----------|
| Claude (claude.ai, API) | ✅ |
| Cursor | ✅ |
| Windsurf | ✅ |
| GitHub Copilot (Workspace) | ✅ |
| Aider | ✅ |
| Any CLI agent with file access | ✅ |

The system is model-agnostic. It works with Claude, GPT-4, Gemini, or any capable model. The protocols are written in plain language, not code — any model that can follow instructions can operate under them.

---

## FAQ

**Do I need to mention `.agent/` in every message?**
No. Once initialized, the agents read the folder at the start of every session automatically. You only need to describe your task.

**What if the AI ignores the `.agent/` folder?**
Some tools require you to include the folder in context explicitly (e.g., `@.agent` in Cursor). After the first session, this happens automatically via the boot sequence. The one-time setup message handles this for the first run.

**Can multiple developers use this on the same project?**
Yes. Commit `.agent/` to your repository. Every developer's AI sessions will share the same project memory, conventions, and active decisions. Conflicts are resolved by the nearest-file rule described in `MEMORY.md`.

**What happens if a session ends mid-task?**
The remaining `ACTION_*.md` files survive the session. The next session reads them and continues from the exact step that was incomplete. Nothing is lost.

**Can I edit `MEMORY.md` manually?**
Yes, and you should when you want to enforce something permanently — a convention, a decision, a preference. The agents will treat it as authoritative.

**What if JINX's plan is wrong?**
Tell the AI the plan is wrong. JINX will reframe and replan. MACHINE will not execute a plan JINX hasn't signed off on.

---

## Design philosophy

machineGPT Agent System is built on one conviction: **AI-assisted development fails not because models aren't capable, but because they have no memory and no structure**.

A capable model working from a fresh context window will produce plausible code that contradicts your architecture, ignores your conventions, and forgets the decision you made three sessions ago. It cannot help it — it has no state.

The `.agent/` system gives the model state. Not by making the model smarter, but by making the context persistent, structured, and authoritative. The agents don't need to be reminded what language you use. They don't need to be told your naming conventions again. They read `MEMORY.md` and they know.

The split between JINX and MACHINE is the second foundation. Thinking and executing are different cognitive modes. A model asked to do both simultaneously will bias toward execution — toward the appearance of progress. Separating them forces the planning step to complete before any code is written. This eliminates an entire class of expensive mistakes: the architecturally wrong implementation that was technically correct.

---

<p align="center">
  <strong>machineGPT</strong> · Agent System · v1.0.0<br>
  <em>Built for engineers who want AI that works like a senior teammate, not a stateless autocomplete.</em>
</p>
