<p align="center">
  <img src="https://img.shields.io/badge/machineGPT-Agent_System-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=" alt="machineGPT Badge" />
  <img src="https://img.shields.io/badge/version-2.1.0-blue?style=for-the-badge" alt="Version Badge" />
  <img src="https://img.shields.io/badge/status-stable-brightgreen?style=for-the-badge" alt="Status Badge" />
</p>

<h1 align="center">Agent — Sovereign Agent Framework</h1>

<p align="center">
  <strong>A stateful, protocol-driven multi-role cognitive architecture designed for elite software engineering.</strong><br>
  <em>Embed the sovereign runtime directly into your repository. Eliminate context decay, abstract leakage, and artificial slop once and for all.</em>
</p>

---

## Table of Contents
1. [The Philosophy and Core Manifesto of Stateful Repositories](#the-philosophy-and-core-manifesto-of-stateful-repositories)
2. [Stateless AI Failure Modes and Pathologies: The Architectural Why](#stateless-ai-failure-modes-and-pathologies-the-architectural-why)
3. [The Multi-Role Check-and-Balance Cognitive Pipeline](#the-multi-role-check-and-balance-cognitive-pipeline)
4. [Exhaustive File Directory Specifications and Inner Schemas](#exhaustive-file-directory-specifications-and-inner-schemas)
   - [RULES.md: The Constitutional Authority and Constraint Classifier Matrix](#1-rulesmd-the-constitutional-authority-and-constraint-classifier-matrix)
   - [MEMORY.md: The Sovereign Persistent State, Known Constraints, and Failure Databases](#2-memorymd-the-sovereign-persistent-state-known-constraints-and-failure-databases)
   - [JINX.md: The Architect Reframing Engine, Tri-Approach Design, and Blast Radius Calculations](#3-jinxmd-the-architect-reframing-engine-tri-approach-design-and-blast-radius-calculations)
   - [MACHINE.md: The 5-Step Execution Loop, Pre-Mutation Checks, and the Breaker Test Mandate](#4-machinemd-the-5-step-execution-loop-pre-mutation-checks-and-the-breaker-test-mandate)
   - [PLAN.md and ACTIVE ACTION_*.md: Active Ephemeral Roadmaps, Transaction Sequence Node Maps](#5-planmd-and-active-action_md-active-ephemeral-roadmaps-transaction-sequence-node-maps)
   - [SCOUT.md: The Intelligence Gathering and Codebase Exploration Engine](#6-scoutmd-the-intelligence-gathering-and-codebase-exploration-engine)
   - [PROTOTYPE.md: The One-Time Discovery and Protocol Bootstrap Compilation](#7-prototypemd-the-one-time-discovery-and-protocol-bootstrap-compilation)
5. [Cognitive Loops and Complex State-Transition Mapping](#cognitive-loops-and-complex-state-transition-mapping)
6. [Anti-Slop Safeguards and Strict Scope Discipline Guidelines](#anti-slop-safeguards-and-strict-scope-discipline-guidelines)
7. [Comprehensive Setup Protocols, IDE Settings, Client Integration, Environment Mapping](#comprehensive-setup-protocols-ide-settings-client-integration-environment-mapping)
8. [Failure Recovery Framework: The Five-Tier Systematic Recovery Protocols](#failure-recovery-framework-the-five-tier-systematic-recovery-protocols)
9. [Context Compression Mechanics, Memory Tiering, and Automatic Archive Rotations](#context-compression-mechanics-memory-tiering-and-automatic-archive-rotations)
10. [System Versioning, Backward Compatibility, and Standard Licensing](#system-versioning-backward-compatibility-and-standard-licensing)

---

## The Philosophy and Core Manifesto of Stateful Repositories

Modern software engineering utilizing Artificial Intelligence is fundamentally handicapped by the stateless paradigm of conversational interfaces. When developers interact with standard autocompletion extensions, online chat models, or generic workspace assistants, they are initiating dialogue with a transient, stateless intelligence. Every individual prompt sequence exists in isolation, blind to surrounding system invariants, historical negotiation logs, style-guide profiles, and localized framework quirks.

The core thesis of machineGPT v2.2.0 is structural: the runtime instructions, operating boundaries, constraints, memory registries, and execution loops of the Artificial Intelligence assistant must be stored directly within the code repository itself under a dedicated `.agent/` directory.

By establishing the codebase as the single source of truth for both the production code and the operational mechanisms of the editing agent, machineGPT establishes a persistent codebase brain. The engineering engagement is converted from high-fatigue, error-prone natural language instructions into a highly disciplined, self-documenting, and self-hardening software delivery pipeline.

---

## Stateless AI Failure Modes and Pathologies: The Architectural Why

Traditional Large Language Model coding systems degrade software architectures over time because of several major structural flaws:

### 1. Cumulative Cognitive Amnesia
As development chats progress across days or sessions, critical context drifts outside the model attention window. The assistant loses track of database schemas, custom interface structures, naming conventions, and constraints. Developers must continuously paste context, re-explain constraints, and correct identical classes of syntax errors, resulting in high cognitive overhead.

### 2. Gradual Architectural Drift
Stateless systems calculate plausibility on a localized token level. They lack systemic orientation, leading to code that duplicates existing helper functions, bypasses established adapter boundaries, introduces circular module imports, or violates clean database separations.

### 3. Cumulative Technical Debt and Technical Larping (AI Slop)
Standard AI assistants tend to maximize visual output to demonstrate competence, often generating unsolicited features, writing mock tests, or embedding fake terminal states. This results in cosmetic noise, unrequested dependencies, and maintenance burden.

### 4. Recursive Regression Loops and Blind Patching
When code compilation fails or static validation blocks deployment, a stateless assistant defaults to superficial patches. It repeatedly applies the same syntactic modifications because it cannot remember the failure vectors identified during previous turns.

---

## The Multi-Role Check-and-Balance Cognitive PipelineTo enforce safety, machineGPT segregates operational privileges into coordinated, specialized role layers. This systemic division of concerns guarantees that planning operations do not directly touch production files, execution operations are preceded by exhaustive reconnaissance, and modifications are verified against active memory sync before validation:

```text
                               ┌──────────────────────────┐
                               │       USER REQUEST       │
                               └─────────────┬────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       RULES.md                                         │
 │   - Constitutes supreme authority.        - Enforces 20 Absolute Prohibitions.         │
 │   - Dictates strict Gate Checks.          - Validates input patterns via Matrix.       │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                           (Protocol Cleared ➔ Handoff to JINX)
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  JINX — THE ARCHITECT                                  │
 │   - Zero raw code permissions.            - Thinks in terms of system boundaries.      │
 │   - Reframes tasks into Approaches A/B/C. - Serializes sprint roadmaps in PLAN.md.     │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                            (Step Activated ➔ Invokes Analyst Scan)
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  SCOUT — THE ANALYST                                   │
 │   - Real-time Memory Consistency Audit.   - Downstream symbol dependency mapping.      │
 │   - Editor-specific search optimization.  - Outputs precise Discovery Reports.         │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                            (Discovery Mapped ➔ Handoff to MACHINE)
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  MACHINE — THE EXECUTOR                                │
 │   - Linear 5-Step Execution Loop.         - Enforces safety validations.               │
 │   - Author of Breaker Tests.              - Operates the Recovery Engine.              │
 │   - Records Debt logs & patterns.         - Cleans up spent ACTION files.              │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                       (Continuous updates of central memory ledger)
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       MEMORY.md                                        │
 │   - Stores constraints & tools.           - Manages the dependency graph.              │
 │   - Holds the Failure Trigger Library.    - Hot/Warm/Cold Tiering & Compressions.      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```��──┐
 │                                       MEMORY.md                                        │
 │   - Stores constraints & tools.           - Manages the dependency graph.              │
 │   - Holds the Failure Trigger Library.    - Hot/Warm/Cold Tiering & Compressions.      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                         │
 │   - Linear 5-Step Execution Loop.         - Enforces static type checking.             │
 │   - Author of Breaker Tests.              - Operates the 5-Tier Recovery Engine.       │
 │   - Records Debt logs & patterns.         - Cleans up spent ACTION files.              │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                       (Continuous updates of central memory ledger)
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       MEMORY.md                                        │
 │   - Stores constraints & tools.           - Manages the dependency graph.              │
 │   - Holds the Failure Trigger Library.    - Hot/Warm/Cold Tiering & Compressions.      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Exhaustive File Directory Specifications and Inner Schemas

The sovereign agent runtime operates inside the `.agent/` folder across several dedicated state and policy files.

### 1. RULES.md: The Constitutional Authority and Constraint Classifier Matrix
RULES.md acts as the system supreme constitution. It overrides all user requests, adjacent agent protocols, and third-party files.

#### Core Constitutional Prohibitions:
1. Every new coding turn must load configuration files in this sequence: RULES.md -> MEMORY.md -> PLAN.md -> ACTIVE_ACTIONS.
2. The agent is strictly forbidden from writing code without an active, JINX-approved `ACTION_*.md` task node file.
3. Partial, incremental writes to files inside the `.agent/` directory are blocked. Modifying state requires the full sequence: Read -> Revise -> Write Entire File.
4. Developers and agents must strictly align with design guidelines, error formats, and architectural invariants logged in MEMORY.md.
5. Code edits must always be preceded by grep and static analysis tracing of the downstream environment.
6. Validation checks (build compilation, static linter, typecheck, target tests) must execute immediately after every filesystem mutation.
7. The logical scope of any task must align with the functional limits defined in the active transaction file and the user prompt.
8. Swallowing execution errors is strictly forbidden. The system must use explicit logging markers or propagate up the call stack.
9. Avoid writing basic comments detailing the "What" of literal code commands. Write exclusively "Why" comments documenting underlying architectural decisions.
10. The use of hardcoded magic numbers, URLs, or environmental parameters in production is blocked. Bind variables to types or immutable configs.
11. Asynchronous calculations must use explicit timeout barriers, capture exceptions, and define recovery pathways.
12. Verify the impact on dependencies and update package manifests before importing new libraries.
13. Security keys, tokens, client secrets, and environment parameters must never be committed to repository history.
14. System chat responses must skip tedious line-by-line code walks. Describe tasks via technical trade-offs and structural bounds.
15. Do not prompt the user for permission to perform routine background tasks such as searching files, running compilation, or linting.
16. The implementation of unprompted features, secondary styling layouts, experimental views, or logging pages is prohibited.
17. Do not touch, rename, or omit core agent files unless executing a complete, coordinated migration.
18. Multi-file batched updates across disparate modules are banned. Modify one target component, validate compilation, and then advance.
19. Respect layer decoupling rules (for example, never call backend service routes inside client rendering blocks).
20. Transactions are declared complete only after post-execution compilation checks succeed and the active ACTION cards are deleted.

RULES.md includes the constitutional Classifier Matrix used to inspect and route inputs:

| Input Classification Pattern | Action Type Response | Handoff Target Path |
| :--- | :--- | :--- |
| Core Ambiguity or Conflict | Suspend processing, output core assumptions, and request clarification. | Interface Layer |
| New Technical Objective | Run the Reframing Engine, map blast radius, create PLAN plus ACTIONs. | JINX.md to PLAN.md |
| Active Transaction Command | Perform investigation, construct custom breaker test, run surgical edits. | MACHINE.md to Codebase |
| Simple System Inquiry | Skip roadmaps. Locate symbol references and explain architectural design. | Output Stream |
| Comprehensive Design Query | Route to JINX. Weigh alternatives, map trade-offs, draft brief. | Architectural Brief |
| Emergency halt | Suspend execution immediately, serialize current progress, and save files. | State Registry |

---

### 2. MEMORY.md: The Sovereign Persistent State, Known Constraints, and Failure Databases
MEMORY.md maps the technical orientation, history, and development guidelines of the repository. It keeps the model aligned over long timelines.

#### Core Modules:
- **System Constraints Index**: Defines environmental constraints (such as port binds, execution limits, and secret proxies).
- **Core Validation Commands**: Declares the terminal CLI syntax to test typecheck correctness, lint formatting, and regression tests.
- **Failure Trigger Registry**: A technical database documenting compiler errors, version collisions, and exceptions, tracking occurrence frequency and exact surgical fixes.
- **Technical Debt Ledger**: A permanent log tracking compromises made during rapid sprints, prioritizing debt into blocks (`P0`), degrades (`P1`), and styling concerns (`P2`).
- **Module Dependency Graph**: A living text diagram mapping import boundaries and interactions across components.

```text
Dependency Map Structure Example:
[UI Component Layer]  ──(imports)──>  [Client Controllers]
         │                                    │
    (denied import)                      (uses model types)
         │                                    │
         ▼                                    ▼
[DB Schema Layer]     <──(queries)───  [API Route Layer]
```

---

### 3. JINX.md: The Architect Reframing Engine, Tri-Approach Design, and Blast Radius Calculations
JINX.md defines JINX operational protocols. As the planning agent, JINX has zero access permissions to write or modify files within the `/src` directory.

#### Operational Responsibilities:
- **The Reframing Engine**: Translates user demands into exactly three distinct pathways:
  1. *Approach A (Minimal Intervention)*: Achieves the objective with minimal code churn, prioritizing immediate system stability and low-overhead pathing.
  2. *Approach B (Scalable Integration)*: Architected for long-term decoupled scaling, utilizing clear design abstractions, patterns of robust composition, and formal interfaces.
  3. *Approach C (Alternative Router)*: An unconventional or divergent design approach (such as changing data structures, selecting streaming over batching, or caching rather than standard querying).
- **Blast Radius Quantification**: Calculates structural impact, outlining modified lines, broken components, and dependencies requiring verification.
- **Sprint Contract Serialization**: Builds the step-by-step logic map within PLAN.md and generates the active step card within ACTION_*.md files.

---

### 4. MACHINE.md: The 5-Step Execution Loop, Pre-Mutation Checks, and the Breaker Test Mandate
MACHINE.md drives the execution engine, focusing on technical precision, exact implementation, and verification.

```
                    ┌─────────────────────────┐
                    │ INVESTIGATE             │
                    │ - Trace target symbols  │
                    │ - Trace data stream     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ IMPLEMENT               │
                    │ - Write surgical edits  │
                    │ - Handle null pointers  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ VALIDATE                │
                    │ - Compile, lint, test   │
                    │ - Write a BREAKER test  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ RECORD                  │
                    │ - Log active debt       │
                    │ - Log failure triggers  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ CLEANUP                 │
                    │ - Delete ACTION card    │
                    │ - Update central PLAN   │
                    └─────────────────────────┘
```

#### Core Operational Phases:
1. **INVESTIGATE**: Search the workspace, locate symbols, trace components, and map pipelines before mutating code. Define Expected State, Actual State, and Divergence Point.
2. **IMPLEMENT**: Author clean, typified, and decoupled logic. Ensure robust error handling and avoid broad, unvalidated edits.
3. **VALIDATE**: Run project lint and build routines. Authors are *required* to write a custom Breaker Test containing edge-case inputs designed to challenge the new code boundaries. If the implementation handles these breaker inputs safely, validation is successful.
4. **RECORD**: Update the roadmap, log new technical debt, and feed newly discovered failure vectors back to MEMORY.md.
5. **CLEANUP**: Remove completed ACTION_*.md files and clean up temporary build assets.

---

### 5. PLAN.md and ACTIVE ACTION_*.md: Active Ephemeral Roadmaps, Transaction Sequence Node Maps
These files represent active, ephemeral states that track progress through the current sprint.

- **PLAN.md**: Includes a measurable binary Definition of Done, the chosen architectural strategy, sequence node maps, and rollback commands for disaster recovery.
- **ACTION_*.md**: Spec files built for consumption by MACHINE, outlining implementation directives, target symbols, exposed signatures, dependency barriers, and binary success assertions.

---

### 6. SCOUT.md: The Intelligence Gathering and Codebase Exploration Engine
Acts as the dedicated intelligence, pattern-matching, and search optimization agent. Invoked immediately prior to and during any active execution phase to map symbol coordinates, trace data stream propagation, verify dependency decoupling boundaries, and optimize query formats for the specific AI editor tools.

---

### 7. PROTOTYPE.md: The One-Time Discovery and Protocol Bootstrap Compilation
A temporary bootstrap utility. On setup, it scans the repository, identifies dependencies and scripting configurations, populates placeholders inside MEMORY.md, calibrates SCOUT's search strategies based on host tools, and then deletes itself.

---

## Cognitive Loops and Complex State-Transition Mapping

The following state diagram details the operational lifecycle of a bugfix or feature task under the machineGPT v2.2.0 framework:

```
[User issues bug report]
          │
          ▼
 RULES.md parsed ➔ Gate Checks pass ➔ Routed to JINX (Reframer Engine)
          │
          ├─► Approach A: Fix locally (evaluated)
          ├─► Approach B: Decouple routing rules (selected)
          └─► Approach C: Alternative bypass (evaluated)
          │
          ▼
 JINX writes PLAN.md & writes `ACTION_*.md`
          │
          ▼
 MACHINE boots ➔ Reads `ACTION_*.md` ➔ Invokes SCOUT to build Search Map
          │
          ├──────────────────────────┐
          ▼ (INVESTIGATE)            ▼ (IMPLEMENT)
 Inspects Discovery Report  Writes surgical fixes to routes
          │                          │
          ├──────────────────────────┘
          ▼
 MACHINE (VALIDATE) ➔ Build/Lint runs
          │
          ├───► Compile OK? ──► YES ──► Write Breaker Test (edge-case parameters)
          │                                  │
          │                                  ├──► Breaker fails? ──► YES (Good) ──► Apply final guard rules
          │                                  └──► Breaker passes? ─► NO ──► Re-write breaker assertions
          │
          └───► Compile FAIL? ─► Run Failure Recovery Framework ➔ Identify error class ➔ Document Pattern
          │
          ▼
 MACHINE (RECORD) ➔ Register technical debt P2 ➔ Update Memory with Route Conventions
          │
          ▼
 MACHINE (CLEANUP) ➔ Delete `ACTION_*.md` ➔ Update PLAN.md step 1 [Done]
          │
          ▼
 [Task successfully resolved. Complete Git status committed.]
```

---

## Anti-Slop Safeguards and Strict Scope Discipline Guidelines

To preserve system aesthetics, machineGPT implements strict rules against superficial "AI Slop."

Standard AI autocomplete extensions and models often generate unnecessary UI elements (such as fake server dashboards, decorative headers, system metrics trackers, or complex logs). These elements add visual clutter and increase maintenance burden.

### Codebase Cleanliness Standards:
1. **No Superficial Dashboards**: Implement only the core UI elements requested. Do not wrap simple layouts in mock status views or command logs.
2. **Strict Scope Discipline**: High-quality UI comes from clean typography, balanced negative space, clear contrast, and deliberate alignment—not unprompted layout features.
3. **Real Data Integrity**: If the user requests external integrations (for example, database storage, email notifications, or authentication profiles), build real, functioning integration pipelines. Do not use random static placeholders.

---

## Comprehensive Setup Protocols, IDE Settings, Client Integration, Environment Mapping

### 1. Repository Installation
Copy the complete `.agent/` folder into your root workspace directory:
```bash
.
├── .agent/
│   ├── RULES.md
│   ├── MEMORY.md
│   ├── JINX.md
│   ├── SCOUT.md
│   ├── MACHINE.md
│   ├── PLAN.md
│   └── PROTOTYPE.md
├── src/
├── package.json
└── README.md
```

### 2. Executing the Bootstrap Sequence
Initiate setup by issuing this instruction within your development environment:
```text
Initialize the machineGPT runtime. Execute .agent/PROTOTYPE.md now.
Scan the repository structure, locate active toolchains, populate MEMORY.md, and inject project intelligence.
Confirm details and delete PROTOTYPE.md upon successful setup.
```

### 3. Integrated Development Environment Profiles

#### Cursor Integration:
- Reference active rules files inside the prompt: `@.agent/RULES.md` and `@.agent/MEMORY.md`.
- Direct the planning tasks to JINX, then instruct the builder to execute: `MACHINE: Implement step 1.`.

#### Windsurf and Cascade Integration:
- The development engine automatically parses the local directory structure. The constraints inside `RULES.md` serve as active guardrails for all code modifications.

#### Claude Desktop and Custom Environments:
- Save `.agent/RULES.md` directly into your customized system instruction sets. The model will refer back to these constitutional constraints on every interaction.

#### Aider Command Line Interface:
Add core state files directly into the active terminal context pool:
```bash
/add .agent/RULES.md
/add .agent/MEMORY.md
```

### 4. AI-Editor Capability Probing, Self-Pruning, and Polymorphic Prompt Adaptation

To maximize transactional efficiency, reduce context-window bloat, and eliminate unnecessary token costs, the architecture includes an active self-improvement and optimization pipeline:

*   **Self-Testing Diagnostic Suite**: Upon setup or re-initialization, the agent runs a series of non-destructive probing tests in the workspace (such as executing dry-run matches, performing mock symbol searches, and evaluating tool response times).
*   **Prompt Self-Pruning**: Before deleting the temporary setup assets (`PROTOTYPE.md`), the system executes a specialized, self-directed refinement filter. It identifies redundancies in `RULES.md` and deletes rules that are handled natively by the discovered editor shell (e.g., Cursor, Windsurf, Aider, or AI Studio). This ensures the constitution remains sharp, condensed, and highly focused.
*   **Polymorphic Mutation**: For early-stage greenfield development where the final system scope is still undefined, JINX and MACHINE maintain a polymorphic, baseline frame of prompt instructions. As successful features and definitive styles are committed, the agents mutatively rewrite themselves—hardcoding successfully proven conventions and removing generic fallback mechanisms.

---

## Failure Recovery Framework: The Five-Tier Systematic Recovery Protocols

When compilation checks fail, or testing suites trigger errors during development, MACHINE bypasses quick-patching and classifies the problem into one of five categories:

```
                            TESTING / BUILD FAILURE
                                       │
                                       ▼
                         Identify Failure Classification
                                       │
 ┌──────────────────────┬──────────────┴───────┬──────────────────────┬──────────────────────┐
 │                      │                      │                      │                      │
 ▼                      ▼                      ▼                      ▼                      ▼
Transient Error       Logic Error        Dependency Error      Ambiguity Error       Scope Out Error
- Flaky network,      - Typo, logic bug,     - Missing packages,   - Conflicting steps,  - Action requires
  or lock file block    regression.            import path error.    vague variables.      edits outside
- Re-run once         - Diagnose trace,      - Scan project locks  - HALT. Do not write  Atoms.
  after clearing.       register failure       and manifests.        code. Prompt user   - Revert changes
                        pattern, fix.                                for details.        & ask JINX to
                                                                                         replan.
```

If any specific error signature fails three times in succession, the system halts execution, logs a diagnostic report, and prompts the developer for guidance.

---

## Context Compression Mechanics, Memory Tiering, and Automatic Archive Rotations

As codebases mature, size increases inside MEMORY.md, which can lead to higher token usage and model attention degradation. machineGPT implements an automated memory tiering and archive rotation protocol:

### The Memory Tiering Sequence:
1. **Hot State** (`MEMORY.md`): Capped at **400 active lines**. This includes active constraints, system conventions, validation commands, recent development logs, and unresolved technical debt.
2. **Warm State**: Stable reference profiles and module structures are compressed into minimalist, high-density key-value notation blocks:
   `auth_engine: {adapter: local-jwt, schema: /db/users, coupling: low, status: stable}`.
3. **Cold State**: Complete historical snapshots of memory are compiled and saved directly to timestamped archive files in `.agent/archives/*.md`.

JINX performs this archive process automatically when it detects that the memory file has crossed limits during boot validation.

---

## System Versioning, Backward Compatibility, and Standard Licensing

All components within `.agent/` include a version footer in their metadata: `*Version: v2.2.0*`. Version mismatches during session starts are flagged and written directly into the system migrations ledger of MEMORY.md.

---

<p align="center">
  <strong>machineGPT v2.2.0</strong> · The Sovereign Codebase Brain · Protocol Stable<br>
  <em>Never explain your code conventions again. Secure your development runtimes.</em>
</p>
