# Copyright 2026 JINX Enterprise Team. All Rights Reserved.
#
# Use of this software is governed by the JINX End User License Agreement (EULA) in the
# LICENSE.md file at the root of this repository: free for personal, educational, research and
# non-commercial use; commercial use needs a separate commercial license.
# ==============================================================================
"""Tool schema access for JINX LLM tool-use declarations.

The declarations themselves are model-facing prose and therefore live in
:mod:`jinx.prompts`, next to the rest of the prompt contract. This module is the
accessor: it hands each caller its own copy so a consumer that mutates the
returned structure cannot corrupt the shared template.
"""

import copy
from typing import Any, Dict, List

from .prompts import TOOL_SCHEMA


def tool_schema() -> List[Dict[str, Any]]:
    """Returns the standardized tool declaration schemas for LLM generation requests.

    These schemas inform the model about supported operations and their
    parameter structures.

    Returns:
        List[Dict[str, Any]]: The array of valid tool declaration schemas.
    """
    return copy.deepcopy(TOOL_SCHEMA)
