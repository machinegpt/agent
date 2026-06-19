[English](README.md) | [Русский](README_RU.md) | [中文](README_ZH.md)

<p align="center">
  <img src="https://img.shields.io/badge/JINX-Enterprise_Agent_Runtime-0F172A?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=" alt="JINX Badge" />
  <img src="https://img.shields.io/badge/version-1.1.8--enterprise-2563EB?style=for-the-badge" alt="Version Badge" />
  <img src="https://img.shields.io/badge/architecture-Process_Isolated_IPC-0D9488?style=for-the-badge" alt="Architecture Badge" />
  <img src="https://img.shields.io/badge/integration-Subprocess_Standard_Streams-059669?style=for-the-badge" alt="Integration Badge" />
</p>

<h1 align="center">JINX — Enterprise Sovereign Agent Runtime Specification</h1>

<p align="center">
  <strong>Technical specification for JINX, an isolated, stateful, protocol-driven cognitive loop designed to operate as a child process inside software engineering host environments.</strong>
</p>

---

## 1. Core Architecture & Inter-Process Communication (IPC)

JINX is an agent runtime designed to run inside a host environment (such as an IDE, command-line editor, or corporate orchestrator). The JINX runtime operates without independent network access or direct external service integrations; all external model invocation, file manipulation, and console execution requests are delegated to the host editor via standard input (`stdin`) and standard output (`stdout`) using structured JSON-RPC communication payloads.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "mainBkg": "#21262d", "nodeBorder": "#8b949e", "nodeTextColor": "#e6edf3"}}}%%
flowchart LR
    classDef sub fill:#161b22,stroke:#30363d,stroke-dasharray: 3 3,color:#c9d1d9;
    classDef state fill:#21262d,stroke:#30363d,stroke-width:2px,color:#e6edf3;
    classDef yaml fill:#161b22,stroke:#30363d,stroke-width:2px,color:#c9d1d9;

    subgraph JINX["JINX Agent Runtime (Subprocess)"]
        direction TB
        SM["State Machine & Protocol<br/>(runner.py)"]:::state
        DB[("Local State<br/>(.agent/JINX.yaml)")]:::yaml
        SM <-->|"Read / Write State"| DB
    end
    style JINX fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph HOST["Host IDE / CLI Editor (Parent Process)"]
        direction TB
        EXE["Tool Execution Engine<br/>(bash_exec / file ops)"]:::sub
        LLM["External LLM Gateway<br/>(API keys & Inference)"]:::sub
    end
    style HOST fill:#0d1117,stroke:#30363d,color:#e6edf3

    SM ==>|"stdout (JSON-RPC Payloads)<br/>jinx_command: llm_generate | bash_exec | file_read | file_write"| HOST
    HOST ==>|"stdin (Response Payloads)<br/>{content: ...} | {output: ...}"| SM
```

### JSON-RPC Communication Specification

When JINX performs an action, it emits a structured JSON object to `stdout` ended with a newline character. The host environment reads this object from the process stream, executes the requested action, and returns the response as a JSON string to JINX's `stdin` ended with a newline.

#### 1. LLM Generation Request (`llm_generate`)
JINX delegates LLM inference to the host.
* **Payload emitted to `stdout`**:
```json
{
  "jinx_command": "llm_generate",
  "params": {
    "system": "System instructions defining the cognitive boundaries.",
    "messages": [{"role": "user", "content": "Round-specific context."}],
    "tools": [
      {
        "name": "bash_exec",
        "description": "Execute a bash or shell script in the environment.",
        "input_schema": {
          "type": "object",
          "properties": {
            "script": {"type": "string", "description": "The script to execute"}
          },
          "required": ["script"]
        }
      },
      {
        "name": "file_read",
        "description": "Read the contents of a file.",
        "input_schema": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the file"}
          },
          "required": ["path"]
        }
      },
      {
        "name": "file_write",
        "description": "Write or overwrite a file with new content.",
        "input_schema": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "The full content to write"}
          },
          "required": ["path", "content"]
        }
      }
    ]
  }
}
```
* **Expected input response on `stdin`**:
```json
{
  "content": [
    {"type": "text", "text": "Analyzing codebase structure."},
    {"type": "tool_use", "id": "call_123", "name": "bash_exec", "input": {"script": "pytest tests/test_core.py"}}
  ]
}
```

#### 2. Shell Command Execution (`bash_exec`)
JINX requests the host to run a shell command.
* **Payload emitted to `stdout`**:
```json
{
  "jinx_command": "bash_exec",
  "tool_use_id": "call_123",
  "params": {
    "script": "pytest tests/test_core.py"
  }
}
```
* **Expected input response on `stdin`**:
```json
{
  "output": "=== 1 passed in 0.05s ==="
}
```

#### 3. File Operations (`file_read` & `file_write`)
JINX delegates file reads and writes to the host.
* **Payload emitted to `stdout` (read)**:
```json
{
  "jinx_command": "file_read",
  "tool_use_id": "call_124",
  "params": {
    "path": "src/core.py"
  }
}
```
* **Expected input response on `stdin` (read)**:
```json
{
  "content": "def run():\n    pass"
}
```

* **Payload emitted to `stdout` (write)**:
```json
{
  "jinx_command": "file_write",
  "tool_use_id": "call_125",
  "params": {
    "path": "src/core.py",
    "content": "def run():\n    return True"
  }
}
```
* **Expected input response on `stdin` (write)**:
```json
{
  "output": "Success"
}
```

---

## 2. Cognitive Loop Execution Protocol

The JINX runtime is governed by an iterative loop executed in discrete phases. Standard state properties are preserved across iterations via `JINX.yaml`.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "mainBkg": "#21262d", "nodeBorder": "#8b949e", "nodeTextColor": "#e6edf3"}}}%%
flowchart LR
    classDef sub fill:#161b22,stroke:#30363d,stroke-dasharray: 3 3,color:#c9d1d9;
    classDef fail fill:#442326,stroke:#f85149,color:#ff7b72;
    classDef pass fill:#1f3b23,stroke:#56d364,color:#85e89d;

    subgraph P1["Phase I: Scope Intake"]
        A["1. Context & Boundary Parsing"]:::sub --> B["2. Write Scope to state.facts"]:::sub
    end
    style P1 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P2["Phase II: Hypothesis Generation"]
        C["3. Register Failure History"]:::sub --> D["4. Evaluate Divergent Strategies"]:::sub
    end
    style P2 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P3["Phase III: Breaker Testing"]
        E["5. Run Boundary Verification"]:::sub --> F["6. Populate requirements Schema"]:::sub
    end
    style P3 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P4["Phase IV: Evaluation & Exit"]
        G{"7. Check Loop Convergence"}:::sub
        G -->|All Pass| H["Success Exit"]:::pass
        G -->|Failed Approaches >= 3| I["Deadlock Trigger"]:::fail
        G -->|Rounds >= 40| J["Hard Cap Trigger"]:::fail
    end
    style P4 fill:#0d1117,stroke:#30363d,color:#e6edf3

    B --> C
    D --> E
    F --> G
```

### Execution Phases

1. **Phase I: Scope Definition & Intake**
   Before starting file mutations, JINX parses the workspace environment and sets the boundaries of the target task. The validated context is written directly to the `state.facts` list in the configuration manifest `JINX.yaml`.

2. **Phase II: Hypothesis Generation & Divergence**
   If a previous round fails, JINX registers the failure reasons under `state.scores`. In subsequent rounds, JINX evaluates alternative strategies. Repeating identical approaches without modification is blocked by protocol rules.

3. **Phase III: Boundary Verification (Breaker Testing)**
   For each strategy, a boundary-testing step ("Breaker Test") must be run. The implementation must be verified against edge cases, exceptional inputs, or performance bounds. The scoring criteria are structured in a binary schema (true/false) under `state.scores[].requirements`.

4. **Phase IV: Multi-Criteria Convergence & Exit**
   After each round, JINX updates the metrics and checks for exit or deadlock conditions:
   * **Exit Condition**: Checked when `round` is greater than or equal to the minimum rounds constraint (`loop.min`) and `exit_ready` is marked as true. Exit occurs if the latest implementation satisfies all core requirements, and no higher score is achieved over the last 3 consecutive rounds.
   * **Deadlock Condition**: Initiated if the round count is greater than or equal to `loop.min` and the same requirements fail on 3 separate approaches. Or if the state is explicitly marked as `deadlock` by the runtime.
   * **Hard Cap**: The execution loop is strictly capped at 40 rounds (`HARD_CAP`), forcing a shutdown to prevent token over-consumption.

### Cognitive Loop Control Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "mainBkg": "#21262d", "nodeBorder": "#8b949e", "nodeTextColor": "#e6edf3"}}}%%
flowchart TD
    classDef start fill:#161b22,stroke:#8b949e,stroke-width:2px,color:#e6edf3;
    classDef process fill:#21262d,stroke:#30363d,stroke-width:2px,color:#c9d1d9;
    classDef decision fill:#161b22,stroke:#30363d,stroke-width:2px,color:#c9d1d9;
    classDef success fill:#1f3b23,stroke:#56d364,stroke-width:2px,color:#85e89d;
    classDef danger fill:#442326,stroke:#f85149,stroke-width:2px,color:#ff7b72;

    A["LLM Response Text"]:::start --> B["parse_state_block()"]:::process
    B --> C{"Find ```yaml/json/yml<br/>code blocks (reversed)"}:::decision
    C -->|"Block found"| D{"≥ 2 state keys<br/>in dict?"}:::decision
    D -->|"Yes"| E["Return parsed dict (update)"]:::process
    D -->|"No"| C
    C -->|"None match"| F["Return None<br/>(state unchanged)"]:::danger

    E --> G["merge_state(jinx, update)"]:::process
    G --> H{"StateBlock.model_validate(update)<br/>Is Valid?"}:::decision
    H -->|"No (ValidationError)"| I["Reject update,<br/>return existing jinx"]:::danger
    H -->|"Yes (OK)"| J["model_dump(exclude_none=True)<br/>→ validated_dict"]:::process
    J --> K{"key in update<br/>AND key in validated_dict?"}:::decision
    K -->|"Yes"| L["s[key] = validated_dict[key]"]:::process
    K -->|"No (null or missing)"| M["Preserve existing s[key]"]:::process
    L --> N["Prune prior_failure<br/>from oldest scores"]:::process
    M --> N
    N --> O["write_jinx(jinx)"]:::process

    O --> P["check_exit()"]:::process
    O --> Q["check_deadlock()"]:::process
    Q --> R["_are_approaches_similar()<br/>Jaccard node+edge sim ≥ 0.7"]:::process
    R --> S{"≥ 3 distinct<br/>clusters per req?"}:::decision
    S -->|"Yes"| T["Deadlock → halt"]:::danger
    S -->|"No"| U["Continue loop"]:::success
```

### Cognitive Process Sequence Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "actorBkg": "#21262d", "actorBorder": "#30363d", "actorTextColor": "#c9d1d9", "actorLineColor": "#30363d", "signalColor": "#8b949e", "signalTextColor": "#c9d1d9", "noteBkgColor": "#161b22", "noteBorderColor": "#30363d", "noteTextColor": "#c9d1d9", "labelBoxBkgColor": "#21262d", "labelBoxBorderColor": "#30363d", "labelTextColor": "#c9d1d9", "loopTextColor": "#c9d1d9", "activationBkgColor": "#21262d", "activationBorderColor": "#30363d"}}}%%
sequenceDiagram
    participant CLI as cli.py (main)
    participant Runner as runner.py (run)
    participant State as state.py
    participant Host as Host Editor (stdin/stdout)

    CLI->>Runner: run(task, min_override)
    Runner->>State: read_jinx()
    State-->>Runner: jinx dict
    Runner->>State: write_jinx(jinx) [init state]

    loop "Outer: rnd < HARD_CAP (40)"
        Runner->>State: read_jinx()
        State-->>Runner: current state

        loop "Inner: tool_depth < TOOL_DEPTH_CAP (20)"
            Runner->>Host: stdout JSON-RPC (llm_generate)
            Host-->>Runner: stdin content_blocks
            alt If tool_use detected
                loop For each tool_use
                    Runner->>Host: stdout JSON-RPC (tool call)
                    Host-->>Runner: stdin tool result
                end
                alt If tool_depth >= TOOL_DEPTH_CAP (20)
                    Note over Runner: Depth Cap Fired (Safety Recovery)
                    Runner->>Host: stdout JSON-RPC (llm_generate with tools=[])
                    Host-->>Runner: stdin content_blocks + state block
                    Note over Runner: Break Inner Loop
                end
            else No tool_use
                Note over Runner: Break Inner Loop
            end
        end

        Runner->>Runner: parse_state_block (last match)
        Runner->>State: merge_state + write_jinx
        alt exit_ready + check_exit
            Runner->>CLI: return (success)
        else deadlock detected or deadlock state
            Runner->>CLI: return (deadlock)
        else HARD_CAP exhausted
            Runner->>CLI: sys.exit(2)
        end
    end
```

---

## 3. State Manifest Specification (`JINX.yaml`)

All cognitive progress, failure logs, tasks, and loop settings are serialized to `JINX.yaml`, located in the isolated `.agent` workspace folder. This structure keeps state metadata out of the project repository root.

```yaml
id: JINX
protocol:
  loop:
    min: 10

state:
  task: "PyJWT RS256 token signing implementation"
  facts:
    - "Workspace root verified"
    - "Configuration schema loaded"
  scores:
    - round: 1
      approach: "PyJWT RS256 token signing implementation"
      prior_failure: null
      requirements:
        compile: true
        unit_tests: false
      pass_count: 1
      all_pass: false
  debt: []
  open: []
  exit_ready: false
  deadlock: false
```

---

## 4. Codebase Component Inventory

The JINX runtime is comprised of the following Python components located in `.agent/` (with core package files under `.agent/src/jinx/`):

* **`jinx.py`** (Entrypoint Bootstrapper, located in `.agent/`):
  Serves as the execution entrypoint. It configures python path environments and delegates parameter passing to the command line parser. Includes an automatic dependency bootstrapper that checks for and automatically installs version-bounded dependencies (`pydantic>=2.0.0`, `pyyaml>=6.0`) into the running environment if missing.
* **`cli.py`** (Argument Parser):
  Parses inputs using Python's `argparse` library. Collects positional argument tasks and the optional `--min` loop iteration override before passing them to the orchestrator.
* **`runner.py`** (Orchestrator):
  Implements the state machine. Contains the main loop logic, processes standard streams to exchange payloads with the host editor, parses structured model output containing markdown YAML state blocks, and evaluates the criteria for exiting and deadlock detection.
* **`state.py`** (Serialization Layer):
  Handles the file operations for the manifest state file `JINX.yaml`. It features:
  * **Dynamic Path Resolution**: Implements a robust multi-tier lookup (via environment variable `JINX_PATH`, development path checks, or recursive upwards directory traversal from the current working directory) to guarantee JINX runs seamlessly both in local repositories and global pip-installed workspaces.
  * **Hardened Models**: Utilizes Pydantic schemas (`ScoreEntry` and `StateBlock`) built with fault-tolerant defaults (e.g. `round=0`, `approach="unspecified"`) to prevent parsing exceptions or state drops even if the LLM omits non-essential metrics from its YAML output block.
* **`tools.py`** (JSON-RPC Helper):
  Defines the available tool schemas (`bash_exec`, `file_read`, `file_write`) exported in LLM generation payloads and formats standardized stdout emissions.

---

## 5. Host Integration & Subprocess Implementation Guide

To integrate JINX, the host editor or corporate orchestrator must spawn the JINX execution command as a child process.

### Spawning Specification
* **Command**: `python .agent/jinx.py "[TASK_DESCRIPTION]"`
* **Process Configuration**: Set `stdout` and `stdin` to `subprocess.PIPE`. Enable text mode (`text=True`) and ensure autoflushing is active.
* **Loop Mechanics**: Parse each line of `stdout` as a JSON object, route the command according to the `jinx_command` property, run the underlying system logic, and write the output back to `stdin` as a single-line JSON string.

### Host Integration Python Example

The following script implements the host-side IPC execution protocol:

```python
import subprocess
import json

def execute_jinx(task_description: str):
    # Spawn JINX as a child process
    process = subprocess.Popen(
        ["python", ".agent/jinx.py", task_description],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True
    )

    try:
        # Stream output line-by-line from the JINX child process
        for line in process.stdout:
            payload = json.loads(line.strip())
            command = payload.get("jinx_command")
            tool_use_id = payload.get("tool_use_id")
            params = payload.get("params", {})

            if command == "llm_generate":
                # Execute corporate LLM generation logic
                # ...
                ai_output = [
                    {"type": "text", "text": "Generated text step."},
                    {"type": "tool_use", "id": "call_01", "name": "bash_exec", "input": {"script": "pytest"}}
                ]
                # Return response JSON back to JINX stdin
                process.stdin.write(json.dumps({"content": ai_output}) + "\n")
                process.stdin.flush()

            elif command == "bash_exec":
                # Run the command on the host environment
                script = params.get("script")
                # ...
                execution_result = "Test suite passed"
                # Return execution response JSON back to JINX stdin
                process.stdin.write(json.dumps({"output": execution_result}) + "\n")
                process.stdin.flush()

            elif command == "file_read":
                # Read local workspace file
                filepath = params.get("path")
                # ...
                file_content = "File content mock"
                process.stdin.write(json.dumps({"content": file_content}) + "\n")
                process.stdin.flush()

            elif command == "file_write":
                # Write to local workspace file
                filepath = params.get("path")
                content = params.get("content")
                # ...
                process.stdin.write(json.dumps({"output": "Success"}) + "\n")
                process.stdin.flush()

    except Exception as e:
        process.kill()
        raise e

    process.wait()
    return process.returncode

if __name__ == "__main__":
    exit_code = execute_jinx("Implement corporate schema update")
    print(f"JINX process terminated with code: {exit_code}")
```

---

## 6. Post-Integration Developer Workflow

Once JINX is successfully launched and the IPC connection is managed by the host editor, the developer's interaction with the firmware operates on an audit-and-intervention model.

### Real-Time Diagnostics
During the execution of JINX, the developer does not need to actively manage standard streams. These are processed entirely by the background IDE wrapper. Instead, the developer monitors progress through the following channels:
1. **State Manifest Auditing**:
   Open `.agent/JINX.yaml` in the editor. This file is updated automatically at the completion of every round. The `state` section acts as a live dashboard:
   * **`facts`**: Tracks all extracted domain properties currently assumed by the agent.
   * **`scores`**: Records the metrics and outcomes of each approach round-by-round, displaying which requirements have passed and what failed.
   * **`debt`**: Lists any trade-offs or shortcuts documented by the agent.
2. **Standard Output Logs**:
   The host wrapper captures JINX's stderr or redirects LLM thought blocks (`{"type": "text"}`) into a native UI tab. This allows real-time viewing of the agent's current cognitive focus.

### Handling Pause and Deadlock Interventions
JINX is designed to automatically halt execution when specific protocol limits are hit, requesting human oversight before proceeding.
* **Deadlock Triggering**:
  If the same requirement fails on 3 distinct strategies, the state changes to `deadlock: true`, and the child process exits with an error status or pauses.
* **Manual Correction Workflow**:
  1. The developer inspects `.agent/JINX.yaml` to identify the failing requirement and approach history.
  2. The developer resolves the blocking issue in the code manually or adjusts the environmental constraints (e.g., correcting database seeds or test environment setups).
  3. The developer can manually modify the `state` properties in `JINX.yaml` to update the facts, debt, or open tasks.
  4. The developer restarts the JINX execution from the CLI via the host command. JINX reads the existing `JINX.yaml` manifest, identifies the historical rounds, and continues the cognitive loop using the updated context.

### Session Verification and Commit
Once the cognitive loop satisfies all exiting criteria, JINX exits cleanly with code `0`.
1. **Review Diff**: The developer inspects the file modifications generated in the repository workspace.
2. **Clear/Archive State**: The developer can safely commit the modified source files. The state metadata inside `.agent/JINX.yaml` remains saved in the isolated workspace directory, ready to serve as context for the next requested task.

---

## 7. machineGPT Pluggable Verification & AI Synthesis Engine

To guarantee 100% compliance, schema conformance, and structural stability of JINX, we feature a professional, unified super test suite located at `scripts/jinx_test.py`. This system integrates static environments and unit regression testing with a fully automated, self-healing dynamic AI Synthesis Engine.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "mainBkg": "#21262d", "nodeBorder": "#8b949e", "nodeTextColor": "#e6edf3"}}}%%
flowchart TD
    classDef sub fill:#161b22,stroke:#30363d,stroke-dasharray: 3 3,color:#c9d1d9;
    classDef standard fill:#21262d,stroke:#30363d,stroke-width:2px,color:#c9d1d9;

    subgraph Engine["AISynthesisEngine (jinx_test.py)"]
        AST["1. Offline AST Parser<br/>(Scans .agent/src/jinx/)"]:::standard
        DIFF["2. API Drift Detector"]:::standard
        GEN["3. Dynamic Test Synthesizer"]:::standard
        AST --> DIFF --> GEN
    end
    style Engine fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph Plugins["tests/enterprise_plugins/"]
        V_CLI["verify_cli.py"]:::sub
        V_TOOLS["verify_tools.py"]:::sub
        V_OTHER["verify_runner.py / verify_state.py / verify_prompts.py"]:::sub
    end
    style Plugins fill:#0d1117,stroke:#30363d,color:#e6edf3

    GEN -->|"Auto-Sync / Code-Preservation"| Plugins
```

### Core Testing Pillars
The test orchestrator runs 9 discrete diagnostic phases grouped into 5 primary verification pillars (or 10 phases and 6 pillars under `--stress` configuration):
1. **Platform & Environment Audit**: Validates runtime constraints, python dependencies (Pydantic, PyYAML, Pytest), and file path resolutions.
2. **Schema & Model Conformance**: Stresses Pydantic models serialization and cycle serialization (state blocks, node schemas, edge schemas) into `.agent/JINX.yaml`.
3. **Graph Similarity & Stress Testing**: Exercises graph mathematical clustering, deadlock clustering, and similarity scaling thresholds under extreme topological conditions.
4. **Pytest Integration Regressions**: Triggers the entire existing pyunit regression test suite natively.
5. **AI-Synthesized Dynamic Verification (Pluggable)**: Dynamically scans the core package structure via Abstract Syntax Trees (AST) and compiles corresponding class, method, and function checks inside `tests/enterprise_plugins/`.
6. **Cognitive Loop Scale & Performance Stress Profiling (`--stress`)**: Simulates 500-node/499-edge ApproachGraphs, profiles full Pydantic cycle and YAML dump/load throughput, and simulates 100-way disjoint clustering deadlock scenarios within tight sub-millisecond budgets.

### Code Preservation Boundary Protocols
Developers and AI agents can extend any dynamic test module under `tests/enterprise_plugins/verify_<module>.py` without worrying about their manual test assertions being overwritten during auto-sync runs. Custom testing logic placed inside designated boundary comments is strictly preserved:
```python
# ==============================================================================
# <CUSTOM_CODE_START>
# Add custom assertions and execution tests below. They will be preserved.
def custom_validation_rules(suite):
    # Your manual custom testing assertions go here
    assert True
# <CUSTOM_CODE_END>
# ==============================================================================
```

### Test Suite CLI Arguments
The test suite can be run from the repository root:
* **Run entire suite**:
  ```bash
  python scripts/jinx_test.py
  ```
* **List discovered core inventory and current dynamic plugin coverage**:
  ```bash
  python scripts/jinx_test.py --ai-list
  ```
* **Force complete AST compilation and synchronization**:
  ```bash
  python scripts/jinx_test.py --ai-sync
  ```
* **Run high-scale cognitive loop stress-testing and microsecond-level performance profiling**:
  ```bash
  python scripts/jinx_test.py --stress
  ```

---

## 8. Integration Protocol for Anthropic Claude Code CLI Host Environment

The JINX architectural specification defines seamless runtime deployment as a managed core within the official **Anthropic Claude Code CLI** console development environment. Under this orchestration scheme, Claude Code assumes the role of the parent orchestration host, translating JINX's logical steps into external services and the local file system.

The orchestration host manages:
1. Interception of incoming user requests.
2. Routing requests to external inference gateways (Claude 3.5 Sonnet via Anthropic API).
3. Executing JINX declarative directives for file read, write, and system command execution.
4. Feeding results back into the JINX cognitive loop via the File-IPC mechanism.

In this repository, `CLAUDE.md` is configured to automatically and unconditionally route all user messages, greetings, and tasks through JINX to ensure the automated orchestration loop works seamlessly. If you prefer to manually invoke JINX, you can edit `CLAUDE.md` to restrict it to explicit requests only.

### Security and Interactive Execution Authorization

By default, the Claude Code CLI security model requires interactive operator confirmation for each file modification and shell command execution. 

> [!TIP]
> **Recommended Secure Setup**: We strongly recommend keeping interactive prompts enabled. This ensures you explicitly review and approve every change JINX proposes before it is executed on your system.

#### OPTIONAL: Non-Interactive Sandbox Mode (For isolated or trust-verified environments)

If you prefer a fully automated, non-blocking cognitive loop (e.g., in a secure, isolated development container, sandboxed virtual machine, or CI/CD runner), you can opt to configure Claude's global settings to auto-approve file edits and specific shell patterns.

> [!CAUTION]
> **CRITICAL SECURITY WARNING**: Enabling `"defaultMode": "acceptEdits"` and auto-approving shell commands disables Claude Code's interactive confirmation prompts. This allows JINX (and any other agent running in this workspace) to read, write, and execute arbitrary commands on your host system without asking for confirmation. Do NOT configure these settings on your main host or in untrusted project environments.

The global settings configuration file is located at the universal path mapped to the active user profile's home directory:
* **Windows OS**: `%USERPROFILE%\.claude\settings.json` (resolves to `C:\Users\<Active_User_Account>\.claude\settings.json` dynamically)
* **macOS / Linux**: `~/.claude/settings.json` (resolves to `/home/<username>/.claude/settings.json`)

#### System Directives to Initialize or Modify the Configuration Profile:

To initialize or edit the security profile under your active user context, execute the appropriate shell command:

* **Windows (PowerShell)**:
  ```powershell
  notepad "$env:USERPROFILE\.claude\settings.json"
  ```
* **Windows (Command Prompt - CMD)**:
  ```cmd
  notepad %USERPROFILE%\.claude\settings.json
  ```
* **macOS / Linux (Terminal)**:
  ```bash
  nano ~/.claude/settings.json
  ```

Insert the following declarative permissions block into the JSON configuration file (wrap in a root `{}` object if the file is being newly created):

```json
{
  "permissions": {
    "defaultMode": "acceptEdits",
    "allow": [
      "Bash(python *)"
    ]
  }
}
```

#### Functional Purpose of Authorization Parameters (for Non-Interactive Mode):
| Parameter | Value Type | Architectural Description & Functional Purpose |
| :--- | :--- | :--- |
| `"defaultMode": "acceptEdits"` | `string` | Configures the host's file sandbox to auto-approve modifications. Allows JINX to perform non-blocking reads and writes of IPC files (`jinx_request.json`, `jinx_response.json`) and target software assets. |
| `"allow": ["Bash(python *)"]` | `array[string]` | Declarative whitelist of terminal command patterns. Permits the host to launch and execute the JINX orchestrator (`python .agent/jinx.py`) without waiting for manual operator approval. |

### Session Launch and Message Routing Procedure

1. Initialize the Claude Code CLI session within the project root directory:
   ```bash
   claude
   ```
2. To start a task with JINX, prefix your request or ask Claude explicitly to run JINX:
   * *“JINX: Add division to calc.py and verify with unit tests”*
   * *“Please run JINX to implement division”*

Following the developer's explicit instruction, the host will bootstrap the JINX orchestrator via `python .agent/jinx.py "[request]"` and transparently coordinate the transactional rounds of the cognitive cycle until the task is fully verified end-to-end.



