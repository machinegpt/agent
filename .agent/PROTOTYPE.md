# PROTOTYPE — Bootstrap Protocol

> Active initialization script. Executes exclusively when MEMORY.md is blank, uninitialized, or undergoes structural regeneration.
> Operating constraints: The loader executes roles of JINX (for analysis, compilation, and dependency extraction) and MACHINE (for command-line search and execution auditing).
> This is a ONE-TIME bootstrap routine. Upon successful integration of the dynamic properties across MEMORY.md, JINX.md, and MACHINE.md, this file is systematically destroyed.

---

## Technical Objective

Establish project-specific context and dynamic protocol parameters by systematically parsing repository assets, compiler configurations, build toolchains, and file organization. Synthesize the extracted metadata and inject the parsed parameters directly into JINX.md, MACHINE.md, and MEMORY.md.

---

## Phase 1: Context Isolation and Discovery

Do not make structural assumptions. Scan files and configuration records across the workspace. Use specific file-viewing tools to inspect configurations instead of relying on generic folder queries.

### 1. Workspace Manifest and Dependency Scans
Locate and parse configuration and manifest files (such as `package.json`, `tsconfig.json`, `gemini-api.config`, `cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, or `requirements.txt`):
- **Core Engineering Stack**: Identify programming languages (TypeScript, JavaScript, Go, Rust, Python, Kotlin, C#) and active frameworks (React, Express, NestJS, Next.js, Android Compose, Django, Spring).
- **Automation and Command Toolchains**: Parse package scripts, execution tasks, and makefiles to extract the exact commands required for **Typecheck / Compilation**, **Static Code Linting**, and **Test Execution**.
- **System Configurations**: Read configuration rules (such as ESLint configs, Prettier layouts, compiler configurations, Vite servers, or Webpack setups).

### 2. Codebase Pattern Mapping
Analyze active source structures across directories (such as `/src`, `/app`, `/lib`, or `/components`):
- **Ecosystem Modules Model**: Verify if the codebase runs on ES Modules (import/export syntax) or CommonJS (require rules).
- **File Hierarchy and Naming Standards**: Deduce structural guidelines (such as kebab-case, PascalCase, or snake_case conventions) across directories, classes, testing suites, and configurations.
- **Error Propagation Routing**: Map how exception handlers behave (such as Express middleware interceptors, global filters, try/catch blocks, or Option/Result types).
- **Downstream Dependency Mapping**: Parse core entrypoints and calculate cross-dependencies to blueprint an initial structural map.

---

## Phase 2: Dynamic Protocol Synthesis

Translate the extracted configuration data into tailored content definitions. If the project directory is a clean greenfield project lacking files, define the respective segments as `—` (New Greenfield Project) rather than leaving brackets or empty sections.

Synthesize three core data structures:

### Block A (Repository Profile and Settings)
- **Metadata Index**: System name, logical purpose, stack definitions, entrypoint path, and project maturity rating (`early` if empty).
- **Styling and Structural Conventions**: Clear rules for camelCase, kebab-case, PascalCase, and architecture bounds.
- **System Command Variables**: Dedicated command strings to execute compilation checks, type auditing, linter assertions, and full-suite testing.

### Block B (Module Boundaries and Invariants)
- **Visual Dependency Graph**: A structured ASCII model illustrating interactions between code modules.
- **Subsystem Invariants**: Concrete structural boundaries and rules to prevent layer breaches (such as prohibiting backend service calls in view templates).

### Block C (Safety Profiles and Guardrails)
- **System Safety Patterns**: Protocols built to avoid failure parameters native to the engineering stack (such as server hydration faults, database connection leaks, unhandled async faults, or thread pool exhaustion).
- **Common Failure Modes**: Groupings of typical runtime blocks (such as missing env variables, path mapping failures, or port bind mismatches).

---

## Phase 3: Surgical Target Injections

Inject the synthesized payload properties directly into JINX, MACHINE, and MEMORY files:

### 1. File Update: MEMORY.md
- Overwrite the profiles, conventions, active environment variables, execution commands, and the visual dependency maps.
- Ensure security and validation segments are fully defined.

### 2. File Update: JINX.md
- Target the precise injection markers:
  ```
  <!-- PROTOTYPE:append JINX_PROJECT_INTELLIGENCE -->
  ...
  <!-- /PROTOTYPE:append -->
  ```
- Replace the block (markers included) with **Block B (Module Boundaries and Invariants)**, supplying the real Dependency Graph and Architectural Invariants table.

### 3. File Update: MACHINE.md
- Target the precise injection markers:
  ```
  <!-- PROTOTYPE:append MACHINE_PROJECT_VALIDATION -->
  ...
  <!-- /PROTOTYPE:append -->
  ```
- Replace the block (markers included) with **Block C (Safety Profiles and Guardrails)**, injecting the real Validation Commands, Safety Patterns, and Common Failure Modes.

---

## Phase 4: Self-Audit and System Purification

1.  **State Audit**: Re-read MEMORY.md, JINX.md, and MACHINE.md in full.
2.  **Completeness Audit**: Confirm that absolutely zero unresolved parameters (such as `<compile/typecheck>` or `<lint>`) remain in active system templates. All segments must contain concrete data or `—` (for greenfield states).
3.  **Active Purge**: Execute a complete deletion of this `PROTOTYPE.md` file using the `delete_file` tool to leave the repository pristine.
4.  **Ready Signal**: Output a concise, highly structured 4-line notification detailing system status:
    ```text
    Project: <Detected Name> | Stack: <Framework list> | State: <Early/Active>
    Validation: <Compile, Lint, Test commands configured>
    Memory Index: Dependencies: <Count> | Invariants: <Count> | Safety Patterns: <Count>
    Agent Protocol Status: JINX v2.1.0 ✓ | MACHINE v2.1.0 ✓ | MEMORY v2.1.0 ✓ | RULES v2.1.0 ✓
    ```

---
*One-time bootstrap instruction sheet. Injects project parameters into active agent configuration.*
*Version: v2.1.0*
