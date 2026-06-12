# PROTOTYPE — Bootstrap

> Read when MEMORY.md is empty. Delete after init.
> You plan like JINX and execute like MACHINE. Only session where this is correct.

## Phase 1: Discover

Do not assume. Find.

```
SCAN:
  manifests → language, deps, scripts, toolchain
  configs → linter, formatter, type checker, bundler
  containers → runtime, services, ports
  env templates → required variables
  source → actual code patterns
```

EXTRACT:
- Language + version
- Dependencies + roles
- Build / test / lint / format commands
- Env vars
- Naming from actual code (not config)
- Import style
- Error handling pattern
- Testing pattern
- Entry points

EMPTY → `—` everywhere. Do not invent.

## Phase 2: Map dependency graph

```
FOR each source file:
  grep imports → what it depends on
  grep exports → what depends on it
BUILD graph
IDENTIFY: entry points, shared modules, boundary layers
```

Write to MEMORY.md `## Dependencies`.

## Phase 3: Find architectural invariants

```
ANALYZE source:
  repeated patterns → conventions
  patterns that break if changed → invariants
  layers that must not cross → boundaries
```

Write to JINX.md `## Architectural Invariants`.

## Phase 4: Discover validation + failure modes

```
FIND commands from manifests
FIND failure modes:
  error handling → what's caught?
  tests → what's tested? what's NOT?
  stack patterns → what frequently breaks?
```

Write to MACHINE.md `## Validation commands` + `## Common failure modes`.

## Phase 5: Write MEMORY.md

Read → compose full → write. All placeholders → real values or `—`.

## Phase 6: Inject into JINX.md

Find `<!-- PROTOTYPE:append JINX_PROJECT_INTELLIGENCE -->`.
Replace with dependency graph + architectural invariants.
Write full file.

## Phase 7: Inject into MACHINE.md

Find `<!-- PROTOTYPE:append MACHINE_PROJECT_VALIDATION -->`.
Replace with validation commands + safety patterns + failure modes.
Write full file.

## Phase 8: Verify

Re-read ALL files. Zero placeholders. Injected sections = real data.

## Phase 9: Plan (if task given)

Write PLAN.md + ACTION files.

## Phase 10: Output + delete

```
Project: <name> | Stack: <detected> | Arch: <detected> | State: <stage>
Validation: <commands or "not configured">
Dependencies: <count> | Invariants: <count> | Failure modes: <count>
Files: MEMORY ✓ | JINX ✓ | MACHINE ✓ | RULES ✓
```

Delete this file.

---
*One-time bootstrap. Injects project-specific intelligence into agent protocols.*
