"""
Jinx Self-Evolution - Compact RT-compatible code improvement.

Integrates with existing Brain systems for learning/healing.
Minimal footprint, bounded operations, async-native.
"""

from .core import (
    CodeHash,
    PatchResult,
    check_syntax,
    check_compat,
    compute_hash,
    apply_patch,
    apply_patch_async,
    record_change,
    eval_quality,
    find_similar,
    index_file,
    init_index,
    get_index,
    on_file_change,
    auto_heal,
)

__all__ = [
    "CodeHash",
    "PatchResult",
    "check_syntax",
    "check_compat",
    "compute_hash",
    "apply_patch",
    "apply_patch_async",
    "record_change",
    "eval_quality",
    "find_similar",
    "index_file",
    "init_index",
    "get_index",
    "on_file_change",
    "auto_heal",
]
