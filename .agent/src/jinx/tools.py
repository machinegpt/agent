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
