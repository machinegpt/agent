"""Regression tests for how the file-IPC loop keeps its history and its settings.

Every JINX round is a fresh process, so whatever the next round needs has to
survive a round trip through the run-state file. These tests play the editor: they
answer one request file at a time and re-enter ``run_file_ipc`` the way a new
process would, instead of calling the handlers with objects held in memory.

Three defects are pinned here:

* the run state dropped the assistant ``tool_use`` that every pending tool call
  hangs on, so after a handful of tool replies each result looked orphaned, the
  request window came out empty, and the depth-limit message never reached the
  model;
* a stale wait on tool results could not re-issue the calls, for the same reason;
* a ``--min`` override given on the command line was forgotten after the first
  state block was accepted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from jinx import learning, prompts, runner, selfpatch


ROUND_PROMPT = "ROUND 1 (at least 2 rounds required before exit is considered)\nCURRENT STATE:\n{}"

STATE_BLOCK = """Round done.

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
"""


def _call(n: int) -> Dict[str, Any]:
    return {"type": "tool_use", "id": "tu_%d" % n, "name": "bash_exec",
            "input": {"script": "echo %d" % n}}


def _orphan_result_ids(messages: List[Dict[str, Any]]) -> List[str]:
    """Ids of tool_result blocks whose tool_use is not earlier in the same list."""
    seen: set = set()
    orphans: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                seen.add(block.get("id"))
            elif block.get("type") == "tool_result" and block.get("tool_use_id") not in seen:
                orphans.append(block.get("tool_use_id"))
    return orphans


class _Host:
    """Stands in for the editor CLI: answers a request file, then resumes JINX."""

    def __init__(self, tmp_path: Path, monkeypatch, self_patch: bool = False) -> None:
        self.request = tmp_path / "jinx_request.yaml"
        self.response = tmp_path / "jinx_response.yaml"
        self.state = tmp_path / "jinx_run_state.yaml"
        monkeypatch.setattr(runner, "REQUEST_PATH", self.request)
        monkeypatch.setattr(runner, "RESPONSE_PATH", self.response)
        monkeypatch.setattr(runner, "RUN_STATE_PATH", self.state)
        monkeypatch.setattr(runner, "_install_signal_handlers", lambda: None)
        monkeypatch.setattr(runner, "_claim_ipc_files", lambda: None)
        monkeypatch.setattr(learning, "LESSONS_PATH", tmp_path / "lessons.yaml")
        monkeypatch.setattr(selfpatch, "SELF_PATCH_ENABLED", self_patch)
        monkeypatch.setattr(selfpatch, "ALLOW_PROTECTED", False)
        monkeypatch.setattr(selfpatch, "BASELINE_DIR", tmp_path / "no_baseline")
        monkeypatch.setenv("JINX_PATH", str(tmp_path / "JINX.yaml"))

    def seed(self, **overrides: Any) -> None:
        """Writes a run state as the first round would have left it."""
        state: Dict[str, Any] = {
            "rnd": 1, "tool_depth": 0,
            "history": [{"role": "user", "content": ROUND_PROMPT}],
            "waiting_for": "llm_generate", "min_rounds": 2, "updated_at": time.time(),
        }
        state.update(overrides)
        self.state.write_text(yaml.safe_dump(state), encoding="utf-8")

    def load_state(self) -> Dict[str, Any]:
        return yaml.safe_load(self.state.read_text(encoding="utf-8"))

    def load_request(self) -> Dict[str, Any]:
        return yaml.safe_load(self.request.read_text(encoding="utf-8"))

    def _answer(self, key: str, payload: Any) -> Dict[str, Any]:
        self.response.write_text(yaml.safe_dump({key: payload}), encoding="utf-8")
        runner.run_file_ipc(None, None)
        return self.load_request()

    def reply(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The model's answer to an llm_generate request; returns the next request."""
        return self._answer("content", blocks)

    def run_tools(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """The host's answer to a tool_calls request; returns the next request."""
        results = [{"tool_use_id": c["id"], "content": "ran %s" % c["id"]}
                   for c in request["calls"]]
        return self._answer("results", results)


class TestToolChainKeepsItsHistory:
    def test_requests_stay_usable_across_a_long_tool_chain(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed()

        for n in range(1, 16):
            tools = host.reply([_call(n)])
            assert tools["type"] == "tool_calls"
            request = host.run_tools(tools)

            assert request["type"] == "llm_generate"
            messages = request["messages"]
            assert messages, "the request after tool reply %d carries no messages" % n
            assert _orphan_result_ids(messages) == [], \
                "a tool_result was sent without the tool_use it answers (reply %d)" % n
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"][0]["tool_use_id"] == "tu_%d" % n, \
                "the newest tool result must be what the model is asked to continue from"

    def test_the_run_state_keeps_the_pending_tool_use(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed()

        host.reply([{"type": "text", "text": "running it"}, _call(1)])

        history = host.load_state()["history"]
        assert history[-1]["role"] == "assistant"
        assert runner._extract_last_tool_calls(history) == [
            {"id": "tu_1", "name": "bash_exec", "params": {"script": "echo 1"}}
        ]

    def test_a_trailing_tool_use_is_persisted_but_not_sent(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": [_call(1)]}]

        runner.write_llm_request(history, 1, 1, 2)

        sent = host.load_request()["messages"]
        assert sent[-1] == {"role": "user", "content": "go"}, \
            "a request must not end on a tool_use that has no result"
        assert not any(isinstance(m["content"], list) for m in sent)
        assert host.load_state()["history"][-1]["content"][0]["id"] == "tu_1"


class TestDepthLimitRequest:
    def test_the_final_request_carries_the_depth_limit_message(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed()

        request: Dict[str, Any] = {}
        for n in range(1, runner.TOOL_DEPTH_CAP + 1):
            request = host.run_tools(host.reply([_call(n)]))

        assert request["type"] == "llm_generate"
        assert request["tools"] == [], "the summary request must not offer tools"
        messages = request["messages"]
        assert messages, "the model was sent no messages at the depth limit"
        assert any(
            isinstance(b, dict) and b.get("text") == prompts.TOOL_DEPTH_CRITICAL_MSG
            for b in messages[-1]["content"]
        ), "the depth-limit message never reached the model"
        assert messages[0]["content"].startswith("[context note]"), \
            "earlier context was elided, so the model has to be told"
        assert _orphan_result_ids(messages) == []

    def test_refusals_at_the_limit_still_end_in_a_message(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch, self_patch=True)
        host.seed(tool_depth=runner.TOOL_DEPTH_CAP - 1)
        forbidden = {"type": "tool_use", "id": "tu_bad", "name": "file_write",
                     "input": {"path": str(selfpatch.SRC_DIR / "selfpatch.py"),
                               "content": "X = 1\n"}}

        request = host.reply([forbidden])

        assert request["tools"] == []
        assert request["messages"][-1]["content"][0]["text"] == prompts.TOOL_DEPTH_CRITICAL_MSG


class TestStaleToolWait:
    def test_a_stale_wait_reissues_the_pending_calls(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed()
        first = host.reply([_call(1), _call(2)])
        monkeypatch.setattr(runner, "_is_run_state_stale", lambda *a, **k: True)

        runner.run_file_ipc(None, None)

        again = host.load_request()
        assert again["type"] == "tool_calls"
        assert again["retry"] is True
        assert again["calls"] == first["calls"]

    def test_a_refused_call_is_not_replayed(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch, self_patch=True)
        host.seed()
        forbidden = {"type": "tool_use", "id": "tu_bad", "name": "file_write",
                     "input": {"path": str(selfpatch.SRC_DIR / "selfpatch.py"),
                               "content": "X = 1\n"}}
        first = host.reply([forbidden, _call(2)])
        assert [c["id"] for c in first["calls"]] == ["tu_2"]
        monkeypatch.setattr(runner, "_is_run_state_stale", lambda *a, **k: True)

        runner.run_file_ipc(None, None)

        assert [c["id"] for c in host.load_request()["calls"]] == ["tu_2"], \
            "the retry must not hand the host the write the guard refused"

    def test_answered_calls_are_never_extracted(self) -> None:
        history = [
            {"role": "assistant", "content": [_call(1), _call(2)]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "refused"}]},
        ]

        assert [c["id"] for c in runner._extract_last_tool_calls(history)] == ["tu_2"]


class TestPersistWindow:
    def test_it_keeps_a_trailing_tool_use(self) -> None:
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": [_call(1)]}]

        assert runner._persist_window(history) == history

    def test_it_is_bounded_and_never_starts_on_an_orphan_result(self, monkeypatch) -> None:
        monkeypatch.setattr(runner, "HISTORY_PERSIST_WINDOW", 4)
        history: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        for n in range(1, 6):
            history.append({"role": "assistant", "content": [_call(n)]})
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_%d" % n, "content": "ok"}]})

        window = runner._persist_window(history)

        assert len(window) <= 4
        assert _orphan_result_ids(window) == []
        assert window[-1] == history[-1]


class TestRequestMessages:
    def test_no_notice_when_nothing_was_elided(self) -> None:
        history = [{"role": "user", "content": "go"}]

        assert runner._request_messages(history) == history

    def test_the_notice_counts_what_the_send_window_left_out(self) -> None:
        history = [{"role": "user", "content": "m%d" % i} for i in range(10)]

        messages = runner._request_messages(history)

        assert messages[0]["content"].startswith("[context note] 4 earlier message(s)")
        assert messages[1:] == history[-runner.HISTORY_WINDOW:]

    def test_big_tool_inputs_are_cut_in_the_request_only(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed()
        big = "x" * 5000
        write = {"type": "tool_use", "id": "tu_w", "name": "file_write",
                 "input": {"path": "a.txt", "content": big}}

        tools = host.reply([write])
        assert tools["calls"][0]["params"]["content"] == big, "the host must get the whole file"
        request = host.run_tools(tools)

        sent = [b for m in request["messages"] if isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "tool_use"][0]
        cap = runner.HISTORY_TOOL_INPUT_CHARS
        assert sent["input"]["path"] == "a.txt"
        assert sent["input"]["content"].startswith("x" * cap)
        assert len(sent["input"]["content"]) < len(big)
        assert "more chars omitted" in sent["input"]["content"]
        kept = [b for m in host.load_state()["history"] if isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "tool_use"][0]
        assert kept["input"]["content"] == big, "a re-issued call needs the full input"

    def test_the_cut_can_be_switched_off(self, monkeypatch) -> None:
        monkeypatch.setattr(runner, "HISTORY_TOOL_INPUT_CHARS", 0)
        messages = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "file_write",
             "input": {"content": "y" * 9000}}]}]

        assert runner._shrink_tool_inputs(messages)[0]["content"][0]["input"]["content"] == "y" * 9000

    def test_shrinking_never_mutates_the_history(self) -> None:
        block = {"type": "tool_use", "id": "t", "name": "file_write",
                 "input": {"content": "z" * 9000, "mode": "w", "lines": 3}}
        messages = [{"role": "assistant", "content": [block]}]

        shrunk = runner._shrink_tool_inputs(messages)

        assert block["input"]["content"] == "z" * 9000
        assert shrunk[0]["content"][0]["input"]["mode"] == "w"
        assert shrunk[0]["content"][0]["input"]["lines"] == 3


class TestMinOverrideSurvivesTheLoop:
    def test_write_llm_request_persists_the_override(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)

        runner.write_llm_request([{"role": "user", "content": "hi"}], 1, 0, 3, min_override=3)

        assert host.load_state()["min_override"] == 3

    def test_a_later_write_keeps_the_stored_override(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        runner.write_llm_request([{"role": "user", "content": "hi"}], 1, 0, 3, min_override=3)

        runner.write_llm_request([{"role": "user", "content": "hi"}], 2, 0, 3)

        assert host.load_state()["min_override"] == 3

    def test_no_override_leaves_no_key(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)

        runner.write_llm_request([{"role": "user", "content": "hi"}], 1, 0, 10)

        assert "min_override" not in host.load_state()

    def test_the_override_still_wins_after_a_state_block_is_merged(
        self, tmp_path, monkeypatch
    ) -> None:
        host = _Host(tmp_path, monkeypatch)
        runner.run_file_ipc("do the thing", 3)
        assert "at least 3 rounds" in host.load_request()["messages"][-1]["content"]

        request = host.reply([{"type": "text", "text": STATE_BLOCK}])

        label = request["messages"][-1]["content"]
        assert label.startswith("ROUND 2 (at least 3 rounds"), \
            "the model's protocol.loop.min replaced the --min override: %r" % label[:60]
        assert host.load_state()["min_rounds"] == 3
        assert host.load_state()["min_override"] == 3

    def test_without_an_override_the_protocol_still_takes_effect(
        self, tmp_path, monkeypatch
    ) -> None:
        host = _Host(tmp_path, monkeypatch)
        runner.run_file_ipc("do the thing", None)

        request = host.reply([{"type": "text", "text": STATE_BLOCK}])

        assert request["messages"][-1]["content"].startswith("ROUND 2 (at least 9 rounds")
        assert "min_override" not in host.load_state()

    def test_the_override_survives_a_whole_tool_chain(self, tmp_path, monkeypatch) -> None:
        host = _Host(tmp_path, monkeypatch)
        host.seed(min_rounds=3, min_override=3)

        for n in range(1, runner.TOOL_DEPTH_CAP + 1):
            host.run_tools(host.reply([_call(n)]))

        state = host.load_state()
        assert state["min_override"] == 3, "a tool or no-tools write dropped the override"
        assert state["min_rounds"] == 3
