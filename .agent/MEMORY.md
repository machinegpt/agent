# MEMORY — Persistent Architectural Ledger and System State

> Persistent codebase brain and structured historical state. Owned and integrated under a unified format.
> Constitutional rules priority check: Directives inside RULES.md take absolute precedence over MEMORY.md. Always parse RULES.md first upon session start.
> Transaction mechanics: Follow the strict state protocol: Read entire ledger -> revise all sections -> write entire file output. Partial string updates or segment omissions are prohibited.
> Staleness limits: When file lines exceed 400 active records OR session count reaches 3 epochs, trigger context-tiering and compression to archives.

---

## SYSTEM-CRITICAL — COMPILATION SENSITIVE

<!-- SECTION:constraints -->
### Known Constraints

**Hard Engine Parameters (Infrastructural and Sandboxed Constraints — Non-Negotiable):**
- System Port Access: Port 3000 serves as the exclusive externally routed port. All dev servers and micro-routers must bind strictly to port 3000.
- Hot Module Replacement (HMR): Disabled (`DISABLE_HMR=true`) on the development plane. Do not attempt HMR reconfiguration.
- Direct-Load Credentials: direct module load execution of secrets is strictly prohibited. Use lazy runtime initialization blocks.
- Client Secrets Leakage: Storing secret variables or keys on browser clients is forbidden. All third-party integrations must be proxied via specialized backend route paths (`/api/*`).

**Architectural Parameters (Project Specific Constraints — Refined during development):**
- —
<!-- /SECTION:constraints -->

<!-- SECTION:validation -->
### System Validation Suite Commands

These terminal commands must execute without warnings prior to transaction completion:
1. **Compilation / Typecheck**: `<typecheck / compilation hook>`
2. **Static Lint Auditing**: `<linter hook>`
3. **Regression Tests Runner**: `<test suite execution hook>`
<!-- /SECTION:validation -->

<!-- SECTION:failure-patterns -->
### Dynamic Failure Pattern Database

Discovered compiler bugs, dependencies collisions, and syntax exceptions:
| Symptom / Pattern Trigger | Error Trace Pattern | Surgical Resolution Strategy | Frequency |
| :--- | :--- | :--- | :--- |
<!-- /SECTION:failure-patterns -->

---

## TECHNICAL SYSTEM PROFILE

<!-- SECTION:project -->
- **System Identifier Name**:
- **Core Product Purpose**:
- **Core Technology Stack**:
- **Architectural Style Model**:
- **Application Boot Entrypoint**:
- **Project Lifecycle Phase**: <early | active-feature | refactor-stage | legacy-stable>
<!-- /SECTION:project -->

---

## CODING GUIDELINES & STANDARDS

<!-- SECTION:conventions -->
- **Subsystem Naming Guidelines**:
  - *Filenames*:
  - *Variables & Procedures*:
  - *Data Types & Signatures*:
  - *Testing Assertions Files*:
- **Exception Catching & Error Routing**:
- **Operational Logging Protocols**:
- **Properties Configuration Schema**:
- **Testing Architecture**:
  - *Framework Engine*:
  - *Test Fixtures Location*:
  - *Runner Invocation Method*:
- **Directory Structural Mapping**:
```text
/
```
<!-- /SECTION:conventions -->

---

## RUNTIME PLATFORM CONTEXT

<!-- SECTION:environment -->
- **Host Dependencies & Setup**:
- **Required Boundary Variables**:
- **Active Downstream Interfaces**:
- **Target Gateway Port**:
<!-- /SECTION:environment -->

---

## HISTORICAL BLUEPRINT DECISIONS

The permanent logical record of core architectural designs and designs rejected:
<!-- SECTION:active-decisions -->
| Decision Date | Selection Made | Design Justification | Rejected Candidate | Current Status |
| :--- | :--- | :--- | :--- | :--- |
<!-- /SECTION:active-decisions -->

---

## REGISTERED SYSTEM TECHNICAL DEBT

Unresolved structural compromises or technical short-cuts:
<!-- SECTION:technical-debt -->
| Datestamp | Debt Description | Severity (P0/P1/P2) | Blast Radius Profile | Elimination Schedule |
| :--- | :--- | :--- | :--- | :--- |

*Severity Index: P0 = blocks release; P1 = structural friction; P2 = minor cosmetic layout mismatch.*
<!-- /SECTION:technical-debt -->

---

## DEVELOPER PREFERENCES INDEX

Personal repository styles, formatting properties, and lint triggers:
<!-- SECTION:user-preferences -->
| Preference Parameter | Creation Date | Driving Rationale |
| :--- | :--- | :--- |
<!-- /SECTION:user-preferences -->

---

## SYSTEM DATA STREAM DEPENDENCY MAP

Functional and modular import interactions calculated during bootstrap:
<!-- SECTION:dependency-graph -->
```text
[Module Producer] ──(exposes)──> [Module Consumer]
```
<!-- /SECTION:dependency-graph -->

---

## LEDGERS & HISTORICAL LOGS

<!-- SECTION:versioning -->
### Architectural Engine Ledger
- **Active Protocol Version**: v2.1.0
- **Boot Validation Rule**: Cross-reference footers on system sessions bootstrap. Mark discrepancies in historical migration records beneath.
<!-- /SECTION:versioning -->

<!-- SECTION:migration-history -->
### Framework Migration History
| Record Timestamp | Previous Version | Target Version | Applied Reflows & Optimizations |
| :--- | :--- | :--- | :--- |
<!-- /SECTION:migration-history -->

<!-- SECTION:directory-index -->
### Active Subdirectory Index
| Path Directory | Purpose / Domain Assignment | Last Touch Timestamp |
| :--- | :--- | :--- |
<!-- /SECTION:directory-index -->

---
*Last synchronized:*
*Updated by: <JINX | MACHINE>*

---
*Version: v2.1.0*