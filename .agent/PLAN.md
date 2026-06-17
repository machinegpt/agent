# PLAN

> Active execution roadmap. Designed and owned by JINX. Executed and verified by MACHINE.
> State operation protocol: Read full file -> revise entire structure -> write full file output. Partial string updates or segment omission are strictly prohibited.

## Goal Specifications

- **Definition of Done**: <Single, measurable, and highly concrete assertion of success, verifiable through a terminal flag or assertion suite>

## Context & Blast Radius Constraints

- **Technical Context**: <Brief description detailing systemic motivation, core variables, and affected architectural components>
- **System Invariants**: <The immutable development guidelines and layer rules that must not be bent during development>

## Strategic Approach & Analysis

- **Selected Pathway**: <Explain the exact implementation strategy selected and why it offers optimal tradeoff margins>
- **Rejected Strategies**: <Document alternative systems analyzed and the specific reasons behind their rejection>
- **Blast Radius Mappings**:
  - *Files scheduled for mutation*:
  - *Files indexed for verification / dependent callsites*:

## Transaction Sequence Node Map

```
  [Step P0] ACTION_*.md    ─────>  [Step P0] ACTION_*.md
                                                │
                                                └───> [Step P1] ACTION_*.md
```

- [ ] **Step 1: <Technical Title>** `[P0]`
  - **File**: `.agent/ACTION_*.md`
  - **Done When**: <Binary criteria verifiable by test suite execution or signature assertion>
- [ ] **Step 2: <Technical Title>** `[P0]`
  - **File**: `.agent/ACTION_*.md`
  - **Done When**: <Binary criteria verifiable by test suite execution or signature assertion>

## Risk Evaluation & Rollback Protocols

- **Primary Failure Risk**: <Slightest systemic vulnerability, race condition, or environmental lock that might block progression>
- **Rollback Routine**: <Concrete step-by-step git commands or file restorations to perform if testing blocks deployment>

## Unplanned Architectural Deviations

| Step ID | Tactical Modification Applied | Root Rationale | Downstream Structural Debt Level |
| :--- | :--- | :--- | :--- |

## Final Audit Outcome

<Fully populated by the MACHINE execution agent upon contract validation and validation logs insertion>
