# MEMORY

> Shared state. Read before action. Update after change.
> Stale deleted. No history. RULES.md overrides all.
> Update: Read → modify → Write full file.
> Compression: >400 lines OR >3 sessions → archive.

## Critical — read every session

<!-- SECTION:constraints -->
### Known Constraints

**Technical (cannot change):**
-

**Conventional (can change with reason):**
-
<!-- /SECTION:constraints -->

<!-- SECTION:validation -->
### Validation commands

1. <compile/typecheck>
2. <lint>
3. <test>
<!-- /SECTION:validation -->

<!-- SECTION:failure-patterns -->
### Failure Patterns

| Trigger | Error | Fix | Freq |
|---------|-------|-----|------|
<!-- /SECTION:failure-patterns -->

## Project

<!-- SECTION:project -->
**Name:**
**Purpose:**
**Stack:** <detected by PROTOTYPE>
**Architecture:** <detected by PROTOTYPE>
**Entry:**
**State:** <early | stable | refactor | legacy>
<!-- /SECTION:project -->

## Conventions

<!-- SECTION:conventions -->
**Naming:** files: | functions: | variables: | types: | tests:
**Error handling:**
**Logging:**
**Config:**
**Testing:** framework: | location: | run:
**Structure:**
```
/
```
<!-- /SECTION:conventions -->

## Environment

<!-- SECTION:environment -->
**Setup:**
**Env vars:**
**Services:**
**Ports:**
<!-- /SECTION:environment -->

## Decisions

<!-- SECTION:active-decisions -->
| Decision | Why | Rejected | Since |
|----------|-----|----------|-------|
<!-- /SECTION:active-decisions -->

## Debt

<!-- SECTION:technical-debt -->
| Debt | Sev | Impact | Since |
|------|-----|--------|-------|

**Sev:** P0=blocks | P1=degrades | P2=cosmetic
<!-- /SECTION:technical-debt -->

## Preferences

<!-- SECTION:user-preferences -->
| Pref | Since |
|------|-------|
<!-- /SECTION:user-preferences -->

## Dependencies

<!-- SECTION:dependency-graph -->
> What imports/calls what. MACHINE updates when new deps found.

```
<module> → <module>
<function> → <function>
```
<!-- /SECTION:dependency-graph -->

## Context

<!-- SECTION:context-tiers -->
**Hot:** <2 sessions, blocking | **Warm:** >2 sessions, reference | **Cold:** archives
**Split:** >400 active → MEMORY + MEMORY_CONTEXT
**Archive:** `MEMORY_ARCHIVE_YYYY-MM-DD.md` | fail → retain full, retry
<!-- /SECTION:context-tiers -->

<!-- SECTION:versioning -->
## Versioning

**Template:** v1.0 | **Boot:** compare → log → proceed with newer
<!-- /SECTION:versioning -->

<!-- SECTION:migration-history -->
## Migration History

| Date | From | To | Changes |
|------|------|----|---------|
<!-- /SECTION:migration-history -->

<!-- SECTION:directory-index -->
## Directory Index

| Path | Purpose | Last touched |
|------|---------|-------------|
<!-- /SECTION:directory-index -->

*Last updated:*
*Updated by: <JINX | MACHINE>*
