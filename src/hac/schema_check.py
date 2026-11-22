"""Local validation of model JSON against the declared schemas.

The pipeline asks providers for structured output against a schema, but the
schema is a request, not a guarantee: a provider can ignore it, a new provider
can half-support it, and a hand-written fake in a test returns whatever it
holds. Before any model output is turned into a covenant specification and
counted into a ratio, it is validated here, locally, against the same schema
the request carried. Output that fails is rejected whole - the caller's
fallback (the rule-based reader) handles the clause instead - rather than
silently defaulted, because a defaulted field is a number nobody computed.

Only the JSON Schema subset this project uses is implemented: ``type``
(including a list with ``"null"``), ``properties``, ``required``,
``additionalProperties``, ``items`` and ``enum``. That is deliberate: the
schemas are declared in this repository, so the validator cannot drift behind
a feature nobody here uses.

The template-plus-schema shape follows arXiv:2510.23990, where a small model
with a template and guaranteed syntactic validity matched larger proprietary
models on contract transformation. That paper works on ISDA CSA derivatives
contracts, not loan covenants; the principle transfers, its numbers do not.
"""

from __future__ import annotations

from typing import Any

MAX_ERRORS = 5


class SchemaError(ValueError):
    """Model output did not match the schema it was requested against."""


def _type_ok(value: Any, want: str) -> bool:
    if want == "null":
        return value is None
    if want == "boolean":
        return isinstance(value, bool)
    if want == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if want == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if want == "string":
        return isinstance(value, str)
    if want == "array":
        return isinstance(value, list)
    if want == "object":
        return isinstance(value, dict)
    return False


def errors(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """All violations of ``schema`` in ``instance``, worst first found."""
    found: list[str] = []

    def fail(message: str) -> None:
        if len(found) < MAX_ERRORS:
            found.append(message)

    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if not any(_type_ok(instance, t) for t in allowed):
            fail(f"{path}: expected {'/'.join(allowed)}, got {_name(instance)}")
            return found

    if "enum" in schema and instance not in schema["enum"]:
        fail(f"{path}: {instance!r} not in {schema['enum']!r}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                fail(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    fail(f"{path}: unexpected key {key!r}")
        for key, subschema in properties.items():
            if key in instance:
                found.extend(errors(instance[key], subschema, f"{path}.{key}"))

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            found.extend(errors(item, schema["items"], f"{path}[{i}]"))

    return found[:MAX_ERRORS]


def _name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def require_valid(instance: Any, schema: dict, what: str) -> None:
    """Raise SchemaError unless ``instance`` matches ``schema``."""
    problems = errors(instance, schema)
    if problems:
        raise SchemaError(f"{what} failed schema validation: " + "; ".join(problems))


_SPEC_TERM_KEYS = {"cat", "quarter", "negate", "const", "related", "ebitda", "max"}


def require_valid_spec(spec: dict, what: str = "covenant spec") -> None:
    """Structural check of the internal specification a ratio is computed from.

    The JSON schema constrains the model's reply; this constrains the cleaned
    internal form, so a bad shape is caught even if a future caller builds a
    spec by hand. A spec with no numerator term can only produce a fabricated
    number, so it is rejected here rather than scored as zero.
    """
    if spec.get("direction") not in ("min", "max"):
        raise SchemaError(f"{what}: direction must be 'min' or 'max'")
    threshold = spec.get("threshold")
    if threshold is not None and not _type_ok(threshold, "number"):
        raise SchemaError(f"{what}: threshold must be a number or null")
    for field in ("numerator", "denominator"):
        terms = spec.get(field)
        if terms is None:
            continue
        if not isinstance(terms, list) or (field == "numerator" and not terms):
            raise SchemaError(f"{what}: {field} must be a non-empty list of terms")
        for i, term in enumerate(terms):
            if not isinstance(term, dict) or not (set(term) & _SPEC_TERM_KEYS):
                raise SchemaError(f"{what}: {field}[{i}] is not a valid term")
    springing = spec.get("springing")
    if springing is not None and not _type_ok(springing.get("above"), "number"):
        raise SchemaError(f"{what}: springing.above must be a number")
