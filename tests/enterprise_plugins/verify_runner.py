# ==============================================================================
# AI-Generated Enterprise Verification Plugin
# Module: jinx.runner
# Generated At: 2026-06-19T16:54:06Z
#
# This file is dynamically managed by the JINX AI Synthesis Engine.
# Public classes and methods are verified automatically.
# Add custom verification logic in the marked block below to prevent deletion.
# ==============================================================================
import sys
import importlib
from jinx_test import VerificationPhase, EnterpriseVerificationSuite

class VerifyRunnerPhase(VerificationPhase):
    @property
    def name(self) -> str:
        return "verify_runner"

    @property
    def title(self) -> str:
        return "Phase AI: Dynamic Verification of jinx.runner"

    def run(self, suite: EnterpriseVerificationSuite) -> bool:
        success = True
        suite.print_badge("Initiating AI-Synthesized Verification for jinx.runner", True)
        
        # Dynamic import of the target module
        try:
            target_module = importlib.import_module("jinx.runner")
            suite.print_badge("Import of jinx.runner: SUCCESS", True)
        except Exception as e:
            suite.print_badge("Import of jinx.runner: FAILED (" + str(e) + ")", False)
            return False

        # --- CLASS VERIFICATIONS ---
        # --- FUNCTION VERIFICATIONS ---
        # Verify Function parse_state_block
        if hasattr(target_module, "parse_state_block"):
            suite.print_badge("Function parse_state_block: PRESENT", True)
        else:
            suite.print_badge("Function parse_state_block: MISSING", False)
            success = False

        # Verify Function check_exit
        if hasattr(target_module, "check_exit"):
            suite.print_badge("Function check_exit: PRESENT", True)
        else:
            suite.print_badge("Function check_exit: MISSING", False)
            success = False

        # Verify Function check_deadlock
        if hasattr(target_module, "check_deadlock"):
            suite.print_badge("Function check_deadlock: PRESENT", True)
        else:
            suite.print_badge("Function check_deadlock: MISSING", False)
            success = False

        # Verify Function get_tool_result_from_editor
        if hasattr(target_module, "get_tool_result_from_editor"):
            suite.print_badge("Function get_tool_result_from_editor: PRESENT", True)
        else:
            suite.print_badge("Function get_tool_result_from_editor: MISSING", False)
            success = False

        # Verify Function request_llm_from_editor
        if hasattr(target_module, "request_llm_from_editor"):
            suite.print_badge("Function request_llm_from_editor: PRESENT", True)
        else:
            suite.print_badge("Function request_llm_from_editor: MISSING", False)
            success = False

        # Verify Function run
        if hasattr(target_module, "run"):
            suite.print_badge("Function run: PRESENT", True)
        else:
            suite.print_badge("Function run: MISSING", False)
            success = False

        # ==============================================================================
        # <CUSTOM_CODE_START>
        # Add custom assertions and execution tests below. They will be preserved.
        try:
            # Create a sequence of 5 failed approaches
            test_entries = [
                {"approach": "A", "requirements": {"req_1": False}},
                {"approach": "B", "requirements": {"req_1": False}},
                {"approach": "C", "requirements": {"req_1": False}},
                {"approach": "D", "requirements": {"req_1": False}},
                {"approach": "E", "requirements": {"req_1": False}},
            ]
            
            # Save original similarity checker
            original_sim = target_module._are_approaches_similar
            
            # Mock similarity to define a transitive chain: A ~ B, B ~ C, C ~ D, D ~ E.
            # Distant pairs like A and C, or A and E are NOT similar.
            def mock_are_approaches_similar(entry1, entry2):
                name1 = entry1.get("approach", "") if isinstance(entry1, dict) else getattr(entry1, "approach", "")
                name2 = entry2.get("approach", "") if isinstance(entry2, dict) else getattr(entry2, "approach", "")
                pair = tuple(sorted([name1, name2]))
                allowed_pairs = {
                    ("A", "B"),
                    ("B", "C"),
                    ("C", "D"),
                    ("D", "E")
                }
                return pair in allowed_pairs or name1 == name2
            
            target_module._are_approaches_similar = mock_are_approaches_similar
            
            try:
                # With transitive chain tracking, A, B, C, D, E should all fall into 1 cluster.
                # Therefore, check_deadlock should return False (as we only have 1 cluster, not >= 3).
                is_deadlock = target_module.check_deadlock(test_entries, min_rounds=1, rnd=5)
                assert not is_deadlock, "Transitive chain of similar approaches was split into too many clusters, triggering a false positive deadlock."
                suite.print_badge("Custom Assert: check_deadlock handles transitive similarity chains correctly", True)
            finally:
                target_module._are_approaches_similar = original_sim
        except Exception as e:
            suite.print_badge(f"Custom Assert: check_deadlock transitive test failed: {e}", False)
            success = False

        # Verify IPC Contract and Slicing Capabilities
        try:
            import io
            original_stdin = sys.stdin
            original_stdout = sys.stdout
            try:
                # Case 1: Smart editor with "sliced: true"
                sys.stdin = io.StringIO('{"output": "line 2", "sliced": true}\n')
                sys.stdout = io.StringIO()
                res, sliced, is_err = target_module.get_tool_result_from_editor("call_1", "file_read", {})
                sys.stdout = original_stdout
                assert res == "line 2"
                assert sliced is True
                assert is_err is False
                suite.print_badge("Custom Assert: get_tool_result_from_editor detects 'sliced': true flag", True)

                # Case 2: Smart editor with "is_sliced: true"
                sys.stdin = io.StringIO('{"output": "line 3", "is_sliced": true}\n')
                sys.stdout = io.StringIO()
                res, sliced, is_err = target_module.get_tool_result_from_editor("call_2", "file_read", {})
                sys.stdout = original_stdout
                assert res == "line 3"
                assert sliced is True
                assert is_err is False
                suite.print_badge("Custom Assert: get_tool_result_from_editor detects 'is_sliced': true flag", True)

                # Case 3: Legacy editor with no slicing flags
                sys.stdin = io.StringIO('{"output": "line 1\\nline 2\\nline 3"}\n')
                sys.stdout = io.StringIO()
                res, sliced, is_err = target_module.get_tool_result_from_editor("call_3", "file_read", {})
                sys.stdout = original_stdout
                assert res == "line 1\nline 2\nline 3"
                assert sliced is False
                assert is_err is False
                suite.print_badge("Custom Assert: get_tool_result_from_editor handles legacy editor with no slicing flags", True)
            finally:
                sys.stdin = original_stdin
                sys.stdout = original_stdout
        except Exception as e:
            suite.print_badge(f"Custom Assert: IPC and slicing verification test failed: {e}", False)
            success = False
        # <CUSTOM_CODE_END>
        # ==============================================================================

        return success
