# ==============================================================================
# AI-Generated Enterprise Verification Plugin
# Module: jinx.runner.parse_state_block
# Generated At: 2026-08-20T19:30:00Z
#
# This file is dynamically managed by the JINX AI Synthesis Engine.
# Add custom verification logic in the marked block below to prevent deletion.
# ==============================================================================
import importlib
from jinx_test import VerificationPhase, EnterpriseVerificationSuite


class VerifyParseStatePhase(VerificationPhase):
    @property
    def name(self) -> str:
        return "verify_parse_state"

    @property
    def title(self) -> str:
        return "Phase AI: Verification of parse_state_block behavior"

    def run(self, suite: EnterpriseVerificationSuite) -> bool:
        success = True
        suite.print_badge("Initiating Verification for parse_state_block", True)

        try:
            mod = importlib.import_module("jinx.runner")
            suite.print_badge("Import of jinx.runner: SUCCESS", True)
        except Exception as e:
            suite.print_badge(f"Import of jinx.runner: FAILED ({e})", False)
            return False

        if not hasattr(mod, "parse_state_block"):
            suite.print_badge("Function parse_state_block: MISSING", False)
            return False
        suite.print_badge("Function parse_state_block: PRESENT", True)

        parse_fn = getattr(mod, "parse_state_block")

        # Flat single-key block should be recognized
        flat = """
        ```yaml
        exit_ready: true
        ```
        """
        try:
            res = parse_fn(flat)
            if not isinstance(res, dict) or res.get("exit_ready") is not True:
                suite.print_badge("parse_state_block: flat single-key not recognized", False)
                success = False
            else:
                suite.print_badge("parse_state_block: flat single-key recognized", True)
        except Exception as e:
            suite.print_badge(f"parse_state_block(flat) raised: {e}", False)
            success = False

        # Nested single-key under 'state' should be recognized
        nested = """
        ```yaml
        state:
          deadlock: true
        ```
        """
        try:
            res = parse_fn(nested)
            if not isinstance(res, dict) or not isinstance(res.get("state"), dict) or res["state"].get("deadlock") is not True:
                suite.print_badge("parse_state_block: nested single-key not recognized", False)
                success = False
            else:
                suite.print_badge("parse_state_block: nested single-key recognized", True)
        except Exception as e:
            suite.print_badge(f"parse_state_block(nested) raised: {e}", False)
            success = False

        # <CUSTOM_CODE_START>
        # Add further checks here if desired.
        pass
        # <CUSTOM_CODE_END>

        return success
