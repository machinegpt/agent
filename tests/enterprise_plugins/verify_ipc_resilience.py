# ============================================================================== 
# AI-Generated Enterprise Verification Plugin
# Module: jinx.runner
# Generated At: 2026-10-18T00:00:00Z
#
# This file follows the project convention for regression checks: behavioural
# coverage lives in the enterprise plugin system, not in ad-hoc pytest files.
# ==============================================================================
import importlib
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from jinx_test import VerificationPhase, EnterpriseVerificationSuite


class VerifyIpcResiliencePhase(VerificationPhase):
    @property
    def name(self) -> str:
        return "verify_ipc_resilience"

    @property
    def title(self) -> str:
        return "Phase AI: Verification of jinx file-IPC resilience"

    def run(self, suite: EnterpriseVerificationSuite) -> bool:
        success = True
        suite.print_badge("Initiating IPC resilience verification for the file-IPC loop", True)

        try:
            target_module = importlib.import_module("jinx.runner")
        except Exception as e:
            suite.print_badge("Import of jinx.runner: FAILED (" + str(e) + ")", False)
            return False

        def _assert(label: str, ok: bool) -> bool:
            suite.print_badge(label, bool(ok))
            return bool(ok)

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            request = tmp_dir / "jinx_request.yaml"
            response = tmp_dir / "jinx_response.yaml"
            state = tmp_dir / "jinx_run_state.yaml"

            with mock.patch.object(target_module, "REQUEST_PATH", request), \
                 mock.patch.object(target_module, "RESPONSE_PATH", response), \
                 mock.patch.object(target_module, "RUN_STATE_PATH", state), \
                 mock.patch.object(target_module, "_install_signal_handlers", lambda: None), \
                 mock.patch.object(target_module, "_claim_ipc_files", lambda: None), \
                 mock.patch.object(target_module, "AGENT_DIR", tmp_dir):
                state.write_text(
                    yaml.safe_dump({
                        "rnd": 1,
                        "tool_depth": 0,
                        "history": [{"role": "user", "content": "ROUND 1\nCURRENT STATE:\n{}"}],
                        "waiting_for": "llm_generate",
                        "min_rounds": 2,
                        "updated_at": 0,
                    }),
                    encoding="utf-8",
                )

                response.write_text(
                    yaml.safe_dump({
                        "content": [
                            {"type": "text", "text": "running it"},
                            {
                                "type": "tool_use",
                                "id": "tu_1",
                                "name": "bash_exec",
                                "input": {"script": "echo 1"},
                            },
                        ]
                    }),
                    encoding="utf-8",
                )
                target_module.run_file_ipc(None, None)
                next_request = yaml.safe_load(request.read_text(encoding="utf-8")) or {}
                if not _assert(
                    "ipc: pending tool_use survives the round trip",
                    next_request.get("type") == "tool_calls"
                    and next_request.get("calls", [{}])[0].get("id") == "tu_1",
                ):
                    success = False

                state.write_text(
                    yaml.safe_dump({
                        "rnd": 1,
                        "tool_depth": 0,
                        "history": [{"role": "user", "content": "ROUND 1\nCURRENT STATE:\n{}"}],
                        "waiting_for": "llm_generate",
                        "min_rounds": 2,
                        "updated_at": 0,
                    }),
                    encoding="utf-8",
                )
                response.write_text(
                    yaml.safe_dump({
                        "content": [
                            {"type": "tool_use", "id": "tu_1", "name": "bash_exec",
                             "input": {"script": "echo 1"}},
                            {"type": "tool_use", "id": "tu_2", "name": "bash_exec",
                             "input": {"script": "echo 2"}},
                        ]
                    }),
                    encoding="utf-8",
                )
                target_module.run_file_ipc(None, None)
                request_after_reply = yaml.safe_load(request.read_text(encoding="utf-8")) or {}
                with mock.patch.object(target_module, "_is_run_state_stale", lambda *args, **kwargs: True):
                    target_module.run_file_ipc(None, None)
                stale_retry = yaml.safe_load(request.read_text(encoding="utf-8")) or {}
                if not _assert(
                    "ipc: stale wait reissues the pending calls",
                    stale_retry.get("type") == "tool_calls"
                    and stale_retry.get("retry") is True
                    and [c["id"] for c in stale_retry.get("calls", [])] == ["tu_1", "tu_2"],
                ):
                    success = False

                fake_jinx: dict = {}
                with mock.patch.object(target_module, "read_jinx", lambda: fake_jinx), \
                     mock.patch.object(target_module, "write_jinx", lambda data: fake_jinx.update(data)), \
                     mock.patch.object(target_module.selfpatch, "SELF_PATCH_ENABLED", False), \
                     mock.patch.object(target_module, "REQUEST_PATH", request), \
                     mock.patch.object(target_module, "RESPONSE_PATH", response), \
                     mock.patch.object(target_module, "RUN_STATE_PATH", state):
                    state.write_text("", encoding="utf-8")
                    response.unlink(missing_ok=True)
                    target_module.run_file_ipc("do the thing", 3)
                    response.write_text(
                        yaml.safe_dump({
                            "content": [{
                                "type": "text",
                                "text": """Round done.

```yaml
id: JINX
protocol:
  loop:
    min: 9
state:
  task: t
  facts:
  - a fact
  scores:
  - round: 1
    approach: first try
    prior_failure: none
    requirements:
      req_a: true
    pass_count: 1
    all_pass: true
  exit_ready: false
  deadlock: false
```
""",
                            }]
                        }),
                        encoding="utf-8",
                    )
                    target_module.run_file_ipc(None, None)
                    next_after_state_block = yaml.safe_load(request.read_text(encoding="utf-8")) or {}
                    label = next_after_state_block.get("messages", [])[-1].get("content")
                    if not _assert(
                        "ipc: min_override survives a merged state block",
                        isinstance(label, str) and "at least 3 rounds" in label,
                    ):
                        success = False

        finally:
            for path in (tmp_dir / "jinx_request.yaml", tmp_dir / "jinx_response.yaml", tmp_dir / "jinx_run_state.yaml"):
                path.unlink(missing_ok=True)
            tmp_dir.rmdir()

        return success
