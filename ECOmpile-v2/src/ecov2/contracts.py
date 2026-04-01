from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Contract:
    name: str
    schema: dict[str, Any]


class ContractError(ValueError):
    pass


def load_contract(name: str, root: Path) -> Contract:
    path = root / "schemas" / f"{name}.schema.json"
    if not path.exists():
        raise ContractError(f"Missing schema: {path}")
    return Contract(name=name, schema=json.loads(path.read_text(encoding="utf-8")))


def validate(contract: Contract, payload: Any, *, where: str = "record") -> None:
    _validate_node(contract.schema, payload, where=where)


def _validate_node(schema: dict[str, Any], payload: Any, *, where: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        _check_type(expected_type, payload, where)
    required = schema.get("required", [])
    if isinstance(required, list):
        for field in required:
            if not isinstance(payload, dict) or field not in payload:
                raise ContractError(f"{where}: missing required field '{field}'")

    properties = schema.get("properties", {})
    if isinstance(properties, dict) and isinstance(payload, dict):
        additional = schema.get("additionalProperties", True)
        if additional is False:
            unexpected = set(payload.keys()) - set(properties.keys())
            if unexpected:
                first = sorted(unexpected)[0]
                raise ContractError(f"{where}: unexpected field '{first}'")
        for key, sub_schema in properties.items():
            if key not in payload:
                continue
            _validate_node(sub_schema, payload[key], where=f"{where}.{key}")

    enum_values = schema.get("enum")
    if enum_values is not None and payload not in enum_values:
        raise ContractError(f"{where}: value {payload!r} not in enum {enum_values!r}")

    min_length = schema.get("minLength")
    if min_length is not None:
        if not isinstance(payload, str) or len(payload) < int(min_length):
            raise ContractError(f"{where}: string shorter than minLength={min_length}")

    minimum = schema.get("minimum")
    if minimum is not None:
        if not isinstance(payload, (int, float)) or float(payload) < float(minimum):
            raise ContractError(f"{where}: value below minimum={minimum}")

    maximum = schema.get("maximum")
    if maximum is not None:
        if not isinstance(payload, (int, float)) or float(payload) > float(maximum):
            raise ContractError(f"{where}: value above maximum={maximum}")

    min_items = schema.get("minItems")
    if min_items is not None:
        if not isinstance(payload, list) or len(payload) < int(min_items):
            raise ContractError(f"{where}: list shorter than minItems={min_items}")

    items_schema = schema.get("items")
    if items_schema is not None:
        if not isinstance(payload, list):
            raise ContractError(f"{where}: expected list for items schema")
        for index, item in enumerate(payload):
            _validate_node(items_schema, item, where=f"{where}[{index}]")


def _check_type(expected_type: Any, payload: Any, where: str) -> None:
    if isinstance(expected_type, list):
        if any(_is_type_match(item, payload) for item in expected_type):
            return
        raise ContractError(f"{where}: expected one of {expected_type!r}, got {type(payload).__name__}")
    if not _is_type_match(expected_type, payload):
        raise ContractError(f"{where}: expected {expected_type!r}, got {type(payload).__name__}")


def _is_type_match(expected_type: Any, payload: Any) -> bool:
    mapping = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }
    if expected_type not in mapping:
        return True
    python_type = mapping[expected_type]
    if expected_type == "number":
        return isinstance(payload, (int, float)) and not isinstance(payload, bool)
    if expected_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    return isinstance(payload, python_type)
