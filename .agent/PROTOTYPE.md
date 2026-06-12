# PROTOTYPE — Agent System Bootstrap

## When this file is read

This file exists only in a workspace that has never been initialized.
If `.agent/MEMORY.md` has no real content yet (all sections are placeholders),
you are in a bootstrap session. Read and follow this file instead of JINX.md or MACHINE.md.

Once initialization is complete and this file is deleted, JINX.md and MACHINE.md
become the sole instructions for all future sessions.

You are acting as both roles during this bootstrap: you plan like JINX and execute like MACHINE.
This is the only session where that is correct.

---

## Step 1: Discover the project

Scan the workspace. Do not assume a stack — find it.

Look for: package manifests, build configs, environment templates, toolchain configs,
and a sample of existing source files.

Extract what actually exists:
- Language and runtime version
- Dependencies and their roles
- Commands for build, test, lint, and format
- Required environment variables
- Naming and code style patterns from existing source

**If the workspace is empty** (no source files, no manifests):
note "new project, no source" and leave the relevant MEMORY.md fields as `—`.
Do not invent a stack. The user will add real content in the next session.

---

## Step 2: Write MEMORY.md

Fill every `<!-- SECTION:* -->` block with what was found in Step 1.
Write the full file — read current → compose in memory → write entire file.
No partial edits.

Replace every placeholder line. If a field has no discoverable value, write `—` explicitly.
A `—` is honest; a blank line is ambiguous.

---

## Step 3: Write JINX.md

Read the existing JINX.md fully.
Append the following section at the end of the file — after the final `---` line.
Do not modify anything above it.

Write the full file.

The section to append:

```
## Project Constraints

> Added during bootstrap. These constraints are fixed — do not re-open without a
> structural reason. Update this section when the project's architecture changes.

**Stack:**
- Language / runtime: <value from MEMORY.md>
- Framework: <value from MEMORY.md>
- Architecture: <value from MEMORY.md>

**Fixed architectural decisions:**
| Decision | Reason | Do not re-open unless |
|----------|--------|-----------------------|
| <decision> | <why it was made> | <what would justify changing it> |

**Planning rules for this project:**
- <rule derived from the stack — e.g. "All state mutations go through the store layer">
- <rule derived from the architecture — e.g. "No direct DB calls from route handlers">
- <add one rule per hard constraint found; leave list empty if none found yet>
```

---

## Step 4: Write MACHINE.md

Read the existing MACHINE.md fully.
Append the following section at the end of the file — after the final `---` line.
Do not modify anything above it.

Write the full file.

The section to append:

```
## Project Validation

> Added during bootstrap. Run these commands before marking any ACTION complete.
> Update this section when tooling changes.

**Validation commands** (run in this order):
1. <exact command to compile or type-check — found in manifests>
2. <exact command to lint>
3. <exact command to run tests>

If no commands were found in the project: write `— not yet configured` and flag it
in MEMORY.md under Known Constraints.

**Safety patterns for this stack:**
- <language/framework-specific pattern — e.g. "Always await async calls, wrap in try/catch">
- <e.g. "Validate all external input at the boundary before passing inward">
- <add one per real constraint found; leave list empty if none found yet>

**Known failure modes in this stack:**
- <e.g. "ORM silently ignores unknown fields — always check schema after migrations">
- <leave list empty if none found yet>
```

---

## Step 5: Verify before finishing

Before writing the summary, re-read all three files and confirm:

- MEMORY.md: no section still contains a placeholder (blank or generic text from the template).
  Every field is either a real value or an explicit `—`.
- JINX.md: `## Project Constraints` section exists at the bottom and contains real values,
  not the angle-bracket placeholders from the template above.
- MACHINE.md: `## Project Validation` section exists at the bottom with real commands
  or explicit `— not yet configured`.

If any check fails, fix it before continuing.

---

## Step 6: Write PLAN.md (only if a task was given)

If the user provided a task alongside this bootstrap:

- Write Goal, Context, Approach in PLAN.md
- Break into steps, each mapped to one `ACTION_*.md` file
- Each ACTION file must have: Task, Context, Done when

If no task was given, leave PLAN.md as-is.

---

## Step 7: Output summary and delete this file

Output this summary with real values filled in:

```
Project:      <name or "new project">
Language:     <language + version or "—">
Validation:   <commands written to MACHINE.md or "— not yet configured">
Architecture: <one line or "—">
State:        <early dev | stable | refactor | legacy>

Files written:
  .agent/MEMORY.md  — sections filled
  .agent/JINX.md    — Project Constraints appended
  .agent/MACHINE.md — Project Validation appended
  .agent/PLAN.md    — <task written | left empty>

Verification:
  MEMORY.md placeholders remaining: <count or "none">
  JINX.md Project Constraints: <present | missing — fix before proceeding>
  MACHINE.md Project Validation: <present | missing — fix before proceeding>
```

Then delete this file. From the next session onward, JINX.md and MACHINE.md are
the only instructions.

---

*One-time bootstrap. No authority over ongoing sessions.*
*Next session: read JINX.md or MACHINE.md, not this file.*
