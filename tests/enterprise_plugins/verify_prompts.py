# ==============================================================================
# AI-Generated Enterprise Verification Plugin
# Module: jinx.prompts
# Generated At: 2026-08-20T19:01:34Z
#
# This file is dynamically managed by the JINX AI Synthesis Engine.
# Public classes and methods are verified automatically.
# Add custom verification logic in the marked block below to prevent deletion.
# ==============================================================================
import sys
import importlib
from jinx_test import VerificationPhase, EnterpriseVerificationSuite

class VerifyPromptsPhase(VerificationPhase):
    @property
    def name(self) -> str:
        return "verify_prompts"

    @property
    def title(self) -> str:
        return "Phase AI: Dynamic Verification of jinx.prompts"

    def run(self, suite: EnterpriseVerificationSuite) -> bool:
        success = True
        suite.print_badge("Initiating AI-Synthesized Verification for jinx.prompts", True)
        
        # Dynamic import of the target module
        try:
            target_module = importlib.import_module("jinx.prompts")
            suite.print_badge("Import of jinx.prompts: SUCCESS", True)
        except Exception as e:
            suite.print_badge("Import of jinx.prompts: FAILED (" + str(e) + ")", False)
            return False

        # --- CLASS VERIFICATIONS ---
        # --- FUNCTION VERIFICATIONS ---
        # Verify Function construct_round_prompt
        if hasattr(target_module, "construct_round_prompt"):
            suite.print_badge("Function construct_round_prompt: PRESENT", True)
        else:
            suite.print_badge("Function construct_round_prompt: MISSING", False)
            success = False

        # ==============================================================================
        # <CUSTOM_CODE_START>
        # Add custom assertions and execution tests below. They will be preserved.
        try:
            # Verify existence of required prompt constants
            assert hasattr(target_module, "MISSING_STATE_WARNING"), "MISSING_STATE_WARNING is missing from prompts.py"
            assert hasattr(target_module, "TOOL_DEPTH_CRITICAL_MSG"), "TOOL_DEPTH_CRITICAL_MSG is missing from prompts.py"
            suite.print_badge("Prompt Constants: PRESENT", True)

            # Verify content of prompt constants
            assert "REQUIRED markdown YAML state block" in target_module.MISSING_STATE_WARNING
            assert "inner tool-calling depth limit" in target_module.TOOL_DEPTH_CRITICAL_MSG
            suite.print_badge("Prompt Constants: CORRECT", True)

            # Verify existence of construct_round_prompt function
            assert hasattr(target_module, "construct_round_prompt"), "construct_round_prompt is missing from prompts.py"
            assert callable(target_module.construct_round_prompt), "construct_round_prompt is not callable"
            suite.print_badge("Function construct_round_prompt: PRESENT", True)

            # Test round prompt constructor without missing state
            test_state = "task: Verify Refactoring\nfacts: []"
            res_normal = target_module.construct_round_prompt(rnd=2, min_rounds=5, state_dump=test_state, missing_state=False)
            expected_normal = f"ROUND 2 (at least 5 rounds required before exit is considered)\nCURRENT STATE:\n{test_state}"
            assert res_normal == expected_normal, f"Round prompt mismatch without missing state warning.\nExpected:\n{expected_normal}\nGot:\n{res_normal}"

            # Test round prompt constructor with missing state
            res_warning = target_module.construct_round_prompt(rnd=2, min_rounds=5, state_dump=test_state, missing_state=True)
            expected_warning = f"{target_module.MISSING_STATE_WARNING}ROUND 2 (at least 5 rounds required before exit is considered)\nCURRENT STATE:\n{test_state}"
            assert res_warning == expected_warning, f"Round prompt mismatch with missing state warning.\nExpected:\n{expected_warning}\nGot:\n{res_warning}"

            # Verify task is NOT duplicated: no separate TASK: line in prompt
            assert "TASK:" not in res_normal, "Redundant TASK: line found in prompt — task should only appear inside state_dump"
            assert "TASK:" not in res_warning, "Redundant TASK: line found in prompt — task should only appear inside state_dump"
            
            suite.print_badge("Function construct_round_prompt: BEHAVIOR VERIFIED", True)

        except Exception as e:
            suite.print_badge(f"Dynamic Prompt Verification Failed: {e}", False)
            success = False
        # <CUSTOM_CODE_END>
        # ==============================================================================

        return success
