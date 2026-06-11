# AGENTS.md — Shared Memory

> Both JINX and MACHINE read and write this file.
> This file is the only persistent state between sessions.
> Read it completely before any action. Update it after any meaningful change.
> Delete stale content immediately — do not annotate history.

---

## Project

**Name:**
**Purpose:**
**Language + version:**
**Framework / stack:**
**Architecture:**
**State:** [ early dev | stable | refactor | legacy ]

---

## Conventions

*Established patterns in this codebase. Follow without inventing alternatives.*

- Naming:
- Error handling:
- Logging:
- Config:
- Testing:
- File structure:

---

## Active Decisions

*Architectural and structural decisions already made. Do not re-open without reason.*

|Decision|Reason|Made by|
|--------|------|-------|
| | | |

---

## Technical Debt

*Known issues. Record when created. Remove when resolved.*

- [ ]

---

## User Preferences

*Durable behavior changes. Updated when user requests them explicitly.*

- 

---

## Memory Protocol

**Before any action:**
Read this file. Walk from root to target path, read every AGENTS.md found.
Nearest file = local contract. Parent file = global rules. Conflict = nearest wins on local details.

**After any meaningful change:**
Update nearest AGENTS.md when change affects: purpose, structure, conventions, decisions, debt, or user preferences.
Update parent AGENTS.md when parent-level structure changes.
Remove stale text. Do not explain history.

**Meaningful change = anything that would confuse a future session reading this cold.**
Small edits with no behavioral impact = no update needed, but always check.

**New project with no AGENTS.md:**
Scan project. Infer language, framework, architecture, conventions.
Create this file with everything known. Create child AGENTS.md for every folder with its own responsibility.
Do this before writing any code.

**End of every task:**
Re-check all changed paths. Update affected files. Remove stale content. Report any file intentionally left unchanged and why.

---

## Directory Index

*Subdirectories with their own AGENTS.md. Update when any child is created, moved, or deleted.*

| Path | Purpose |
|------|---------|
| | |

---

*Last updated:*
*Updated by: [ JINX | MACHINE ]*
