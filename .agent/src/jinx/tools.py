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
"""Helper functions for formatting and emitting JSON-RPC tool schemas and payloads."""

import json
import logging
from typing import Any, Dict, List

# Setup a dedicated stderr logger for internal tool logs
logger = logging.getLogger("jinx.tools")


def emit_command(command: str, params: Dict[str, Any]) -> None:
    """Emits a structured JSON-RPC command payload to stdout.

    The host editor/IDE intercepts this line from stdout and executes the
    command in its local sandboxed runtime, feeding results back to stdin.

    Args:
        command (str): The name of the IPC-delegated command.
        params (Dict[str, Any]): The operational parameters for the command.
    """
    payload: Dict[str, Any] = {
        "jinx_command": command,
        "params": params
    }
    try:
        # Strict stdout emissions must be single-line JSON with an explicit newline
        print(json.dumps(payload), flush=True)
    except OSError as e:
        logger.error("Failed to write JSON-RPC command output to stdout stream: %s", e)


def request_bash_execution(script: str) -> None:
    """Requests the host editor/IDE to run a bash or shell script.

    Args:
        script (str): The raw terminal script content to execute.
    """
    emit_command("bash_exec", {"script": script})


def request_file_read(filepath: str) -> None:
    """Requests the host editor/IDE to read file contents into standard memory.

    Args:
        filepath (str): The absolute or workspace-relative path of the target file.
    """
    emit_command("file_read", {"path": filepath})


def request_file_write(filepath: str, content: str) -> None:
    """Requests the host editor/IDE to write or overwrite a file with contents.

    Args:
        filepath (str): The absolute or workspace-relative path of the target file.
        content (str): The full text payload to be written to the destination file.
    """
    emit_command("file_write", {"path": filepath, "content": content})


def tool_schema() -> List[Dict[str, Any]]:
    """Returns the standardized JSON-RPC schemas of JINX operational tools.

    These are passed in LLM generation requests to inform the model about
    supported operations.

    Returns:
        List[Dict[str, Any]]: The array of valid tool declaration schemas.
    """
    return [
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
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Optional 1-indexed starting line to read (inclusive)"},
                    "end_line": {"type": "integer", "description": "Optional 1-indexed ending line to read (inclusive)"}
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

