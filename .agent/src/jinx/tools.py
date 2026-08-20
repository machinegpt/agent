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
"""Tool schema definitions for JINX LLM tool-use declarations."""

from typing import Any, Dict, List


def tool_schema() -> List[Dict[str, Any]]:
    """Returns the standardized tool declaration schemas for LLM generation requests.

    These schemas inform the model about supported operations and their
    parameter structures.

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
                    "script": {
                        "type": "string",
                        "description": "The script to execute"
                    }
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed starting line to read (inclusive)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed ending line to read (inclusive)"
                    }
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }
    ]
