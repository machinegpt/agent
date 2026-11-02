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
import os
import sys

from . import export, runner, sessions, team
from .state import AGENT_DIR

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
    parser.add_argument(
        "--session",
        default=None,
        help="Session to resume (several JINX sessions can run in one project at the same time). "
             "Every call prints its id on the [JINX_SESSION] line; JINX_SESSION in the environment "
             "does the same. With a task, names the new session. Not needed while JINX recognises "
             "the calling editor (the [JINX_SESSION] line says when)."
    )
    parser.add_argument(
        "--join",
        default=None,
        metavar="SESSION",
        help="Start a new session as a helper of another one: it joins that session's team (mail to 'room' "
             "reaches both) and is briefed once about what the other has done. The task is optional."
    )
    parser.add_argument(
        "--peek",
        nargs="?",
        const="*",
        default=None,
        metavar="SESSION",
        help="Look at the other sessions without joining: with no id, one line per session; with an id, "
             "what it does and which rounds changed which files. Changes nothing."
    )
    parser.add_argument("--diff", action="store_true", help="With --peek <id>: also show the changes themselves.")
    parser.add_argument("--full", action="store_true",
                        help="With --peek <id>: no line limits (the output is still cut at 400,000 characters).")
    parser.add_argument(
        "--export",
        nargs="*",
        default=None,
        metavar="SESSION",
        help="Collect the rounds of the named sessions (all of them with none named) into one .mbox that "
             "`git am` applies, one commit per round, and a .yaml that records them. Changes nothing in the project."
    )
    parser.add_argument("--out", default=None, help="With --export: the folder to write to (default: .agent/export).")

    args = parser.parse_args()

    if args.peek is not None:
        sys.exit(team.peek_main(sys.argv[1:], AGENT_DIR))
    if args.export is not None:
        sys.exit(export.export_main(sys.argv[1:], AGENT_DIR))

    task_str: str = " ".join(args.task).strip()

    if args.join and not (args.ipc == "file" and sessions.enabled()):
        logger.error("--join needs several-session mode: file IPC, with JINX_SESSIONS not set to 0.")
        sys.exit(1)

    if args.ipc == "file" and sessions.enabled():
        # Every session has a directory of its own for the request, the response, the run state
        # and the manifest, so a second editor cannot overwrite the first one's files. The
        # runner reads its paths from the module, which is why they are rebound here.
        try:
            ctx = sessions.open_session(task_str or None, args.session or os.environ.get("JINX_SESSION") or None,
                                        join=args.join)
        except sessions.SessionError as e:
            logger.error("%s", e)
            sys.exit(1)
        runner.bind_session(ctx)
        print(sessions.describe(ctx), flush=True)

    is_resuming = (args.ipc == "file") and runner.RUN_STATE_PATH.exists()

    if not task_str and not is_resuming:
        logger.error("No task argument provided. Please specify a task description.")
        sys.exit(1)

    try:
        runner.run(task_str if task_str else None, args.min, ipc_mode=args.ipc)
    except KeyboardInterrupt:
        logger.warning("\n[JINX] Session execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical("Unexpected runtime exception encountered: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
