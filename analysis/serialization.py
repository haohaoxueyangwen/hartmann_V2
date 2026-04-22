from __future__ import annotations

import json
from typing import Any

import numpy as np


def to_builtin_serializable(obj: Any, *, stringify_dict_keys: bool = False) -> Any:
    """Recursively convert NumPy-heavy objects into JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {
            (str(k) if stringify_dict_keys else k): to_builtin_serializable(
                v,
                stringify_dict_keys=stringify_dict_keys,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [to_builtin_serializable(v, stringify_dict_keys=stringify_dict_keys) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_json_serializable(
    payload: Any,
    output_path: str,
    *,
    indent: int = 2,
    stringify_dict_keys: bool = False,
) -> None:
    """Serialize a Python/NumPy payload to JSON after converting NumPy scalars/arrays."""
    with open(output_path, "w") as fp:
        json.dump(
            to_builtin_serializable(payload, stringify_dict_keys=stringify_dict_keys),
            fp,
            indent=indent,
        )
