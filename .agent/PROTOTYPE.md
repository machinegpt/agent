# PROTOTYPE — Bootstrap Protocol

> Active initialization script. Runs exclusively when MEMORY.md is blank, uninitialized, or undergoes structural regeneration.
> Operating roles: Executes as JINX (analysis, compilation, dependency mapping) and MACHINE (search, execution auditing).
> This is a ONE-TIME bootstrap routine. Upon integration of project parameters in MEMORY.md, JINX.md, and MACHINE.md, this file is deleted.

---

## Technical Objective

Establish project-specific context and dynamic protocol parameters by parsing repository assets, compiler configurations, build toolchains, and file structures. Synthesize this metadata and inject the parsed parameters directly into JINX.md, MACHINE.md, and MEMORY.md.

---

## Phase 1: Context Isolation and Discovery

Inspect configurations instead of relying on broad folder queries.

### 1. Workspace Manifest and Dependency Scans
Locate and parse manifest files, dependency declarations, and configuration matrices in the workspace root:
- **Core Engineering Stack**: Dynamically identify programming languages, build managers, compilers, and active frameworks native to this workspace.
- **Automation Commands**: Parse package configurations, workspace manifests, makefiles, or execution scripts to extract exact commands for **Compilation**, **Linting**, and **Testing**.
- **System Configurations**: Inspect formatter settings, compiler boundaries, workspace settings, and local dev execution parameters.

### 2. Codebase Pattern Mapping
Analyze the primary directories, modules, and file groupings:
- **Ecosystem Model**: Determine the packaging format, module resolution protocols, or architectural bindings.
- **Naming Standards**: Deduce casing and naming patterns (variables, folders, modules, procedures, test files) used in the workspace.
- **Error Propagation**: Map standard runtime exception patterns, monads, or local recovery flows.
- **Dependency Paths**: Map entry points, identify core system boundaries, and trace direct modular coupling.

### 3. Editor Tool Diagnostic Trials
Run minimal, safe test behaviors to analyze execution tooling limits:
- **Search Latency & Grep Options**: Run a safe, scoped search to check path resolution, folder exclusions, and match formatting, calibrating SCOUT to avoid overhead.
- **Edit Execution & Indent Mechanics**: Run a dry-run edit to verify substring matching, line ranges, or patch mechanical behaviors, profiling spacing and line-termination conventions.
- **Redundant Validation Checks**: Test if mutations trigger excessive linter or compiler runs, configuring parameters to minimize latency.
- **Model Attention Baseline**: Evaluate response delay on rich layouts, establishing strict boundaries.

Discoveries must be compiled and injected into `MEMORY.md` under respective sections.

---

## Phase 2: Dynamic Protocol Synthesis

Translate extracted config data into tailored properties. If the repository is a clean greenfield project lacking files, define targets as `—` (New Greenfield Project) rather than leaving template brackets.

Synthesize three core blocks:

### Block A (Project Settings and Manifest)
- **Metadata Index**: System identifier name, core purpose, tech stack, entrypoint, and maturity.
- **Conventions**: Naming rules (kebab-case, camelCase, PascalCase) and folder architecture bounds.
- **Commands**: Explicit terminal execution strings for typechecks, linter checks, and test runner executions.

### Block B (Module Boundaries and Structures)
- **Ascii Dependency Map**: Structured ASCII tree outlining module interactions and import directions.
- **System Invariants**: Layer boundary rules (e.g., separating database entity imports from template-view loaders).

### Block C (Safety Profiles and Guardrails)
- **Safety Patterns**: Protocols to avoid stack-native failures (e.g., hydration mismatches, DB leaks, unhandled rejections).
- **Failure Patterns**: Common stack compilation blocks or environment exceptions (e.g., secret load failures, host bindings).

---

## Phase 3: Surgical Target Injections

Inject payloads directly into the agent runtime files:

### 1. Update MEMORY.md
- Overwrite profiles, environments, conventions, commands, and dependency graphs in respective sections.

### 2. Update JINX.md
- Target injection markers:
  ```
  <!-- PROTOTYPE:append JINX_PROJECT_INTELLIGENCE -->
  ...
  <!-- /PROTOTYPE:append -->
  ```
- Replace the block (including markers) with **Block B (Module Boundaries and Structures)**.

### 3. Update MACHINE.md
- Target injection markers:
  ```
  <!-- PROTOTYPE:append MACHINE_PROJECT_VALIDATION -->
  ...
  <!-- /PROTOTYPE:append -->
  ```
- Replace the block (including markers) with **Block C (Safety Profiles and Guardrails)**.

---

## Phase 4: Active Refinement and Self-Mutation

Before deleting setup assets, perform a self-directed optimization over the active `.agent/*` configuration frameworks:

### 1. RULES.md Prompt Pruning
- **Redundancy Pruning**: Match parsed editor capabilities. Strip verbose tool parameter explanations or coordinates handled natively by the IDE structure.
- **Attention Focus Compression**: Condense constitutional rules in `RULES.md` into highly declarative, token-efficient imperative statements to avoid cognitive drift.

### 2. Guarded Self-Mutation Mechanics
When JINX, MACHINE, and SCOUT optimize or adapt their instructions over time, the system must assert compliance with these absolute boundaries:
- **Core Loop Preservation**: You are strictly forbidden from disabling, softening, or omitting any of the 20 Absolute Prohibitions or the distinct roles of Planning (JINX), Execution (MACHINE), and Analysis (SCOUT).
- **Structure and Version Validation**: Self-mutated files must maintain syntactical layout, proper markdown rendering, and the core version footer `*Version: v2.2.0*`.
- **Factual Ledger Reconciliation**: Any optimization or tailored protocol update must log the driving rationale underneath the user-preferences matrix in `MEMORY.md`.

### 3. Purification and Readiness Run
1. **Self-Audit**: Re-read MEMORY.md, JINX.md, MACHINE.md, and SCOUT.md.
2. **Completeness Verification**: Confirm zero unpopulated templates (such as `<compile/typecheck>`) remain. Early stack values must read `—`.
3. **Execution Cleansing**: Delete this `PROTOTYPE.md` file using the `delete_file` tool.
4. **Signal Ready state**: Output a concise, 4-line confirmation block:
   ```text
   Project: <Name> | Stack: <Frameworks> | State: <Maturity>
   Validation: <Compile, Lint, Test commands mapped>
   Memory Index: Dependencies: <Count> | Invariants: <Count> | Safety Patterns: <Count>
   Agent Protocol Status: JINX v2.2.0 ✓ | MACHINE v2.2.0 ✓ | MEMORY v2.2.0 ✓ | RULES v2.2.0 [Self-Pruned] ✓ | SCOUT v2.2.0 ✓
   ```

---
*One-time bootstrap instruction sheet. Injects project parameters into active agent configuration.*
*Version: v2.2.0*
