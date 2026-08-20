# Copyright 2026 JINX Enterprise Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Command-line parsing interface and runner delegation module for JINX."""

import argparse
import logging
from pathlib import Path
import sys

from .runner import run

# Setup a dedicated stderr logger for internal CLI logs
logger = logging.getLogger("jinx.cli")


def main() -> None:
    """Parses command-line arguments and invokes the JINX cognitive loop."""
    # Ensure system logs go to stderr so we do not pollute stdout JSON-RPC channel
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stderr
    )

    parser = argparse.ArgumentParser(
        description="JINX Sovereign Agent CLI (Firmware Mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="The specific task description or objective for the agent to resolve."
    )
    parser.add_argument(
        "--min",
        type=int,
        default=None,
        help="Override the minimum loop iteration rounds (loop.min) specified in JINX.yaml."
    )
    parser.add_argument(
        "--ipc",
        choices=["file", "rpc"],
        default="file",
        help="Communication protocol channel (file-based state machine or standard JSON-RPC stream)."
    )

    args = parser.parse_args()

    task_str: str = " ".join(args.task).strip()
    
    run_state_path = Path(__file__).resolve().parent.parent.parent / "jinx_run_state.yaml"
    is_resuming = (args.ipc == "file") and run_state_path.exists()

    if not task_str and not is_resuming:
        logger.error("No task argument provided. Please specify a task description.")
        sys.exit(1)

    try:
        run(task_str if task_str else None, args.min, ipc_mode=args.ipc)
    except KeyboardInterrupt:
        logger.warning("\n[JINX] Session execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical("Unexpected runtime exception encountered: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
