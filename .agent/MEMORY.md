# MEMORY

> Shared state for JINX and MACHINE.
> Read completely before any action. Update after every meaningful change.
> Delete stale content — do not annotate history.
> **Update rule: always Read → modify in memory → Write full file.**

---

<!-- SECTION:project -->
## Project

**Name:**
**Purpose:**
**Language + version:**
**Framework / stack:**
**Architecture:**
**Entry points:**
**State:** [ early dev | stable | refactor | legacy ]
<!-- /SECTION:project -->

---

<!-- SECTION:conventions -->
## Conventions

**Naming:**
- Files:
- Functions:
- Variables:
- Types / interfaces:
- Tests:

**Error handling:**

**Logging:**

**Config:**

**Testing:**
- Framework:
- Location:
- Coverage expectations:

**File structure:**
```
/
```
<!-- /SECTION:conventions -->

---

<!-- SECTION:environment -->
## Environment

**Local setup:**
**Required env vars:**
**External services:**
**Ports / endpoints:**
<!-- /SECTION:environment -->

---

<!-- SECTION:active-decisions -->
## Active Decisions

> Architectural and structural decisions already made. Do not re-open without reason.

| Decision | Reason | Alternatives rejected | Made by |
|----------|--------|-----------------------|---------|
| | | | |
<!-- /SECTION:active-decisions -->

---

<!-- SECTION:conflict-resolution -->
## Conflict Resolution Protocol

> How contradictions between state files are resolved during boot.

When the same fact appears in multiple files with different values:

1. **Nearest file wins** — priority order (highest → lowest): `ACTION_*.md` → `PLAN.md` → `MEMORY.md`
2. The winning value is written back into `MEMORY.md` to resolve the contradiction.
3. The losing value is discarded — not archived.

Rationale: ACTION files are the most recent intent; MEMORY.md is the most stable baseline. More specific/recent always overrides more general/older.
<!-- /SECTION:conflict-resolution -->

---

<!-- SECTION:known-constraints -->
## Known Constraints

> Hard limits that shape every implementation decision. Non-negotiable unless explicitly changed.

-
<!-- /SECTION:known-constraints -->

---

<!-- SECTION:technical-debt -->
## Technical Debt

> Record when created. Remove when resolved.

| Debt | Created at | Impact | Owner |
|------|-----------|--------|-------|
| | | | |
<!-- /SECTION:technical-debt -->

---

<!-- SECTION:user-preferences -->
## User Preferences

> Durable behavior changes. Updated only when user requests explicitly.

-
<!-- /SECTION:user-preferences -->

---

<!-- SECTION:directory-index -->
## Directory Index

> Subdirectories with distinct responsibility. Update when structure changes.

| Path | Purpose | MEMORY.md |
|------|---------|-----------|
| | | [ yes \| no ] |
<!-- /SECTION:directory-index -->

---

*Last updated:*
*Updated by: [ JINX | MACHINE ]*
