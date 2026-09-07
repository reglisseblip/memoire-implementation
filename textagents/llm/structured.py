"""Turn a pydantic model into a wire schema a strict endpoint accepts, and validate the answer.

pydantic emits enums as `$defs` plus `$ref`, strict structured-output mode refuses a `$ref` that
carries sibling keys, and the provider prunes `minimum` and `maximum` from the wire schema. The
references are therefore inlined here and the numeric bounds survive in the field descriptions,
which pydantic re-imposes on receipt.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError


def strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Wire schema with every reference inlined and every field required."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                target = inline(defs.get(name, {}))
                siblings = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **siblings}
            return {key: inline(value) for key, value in node.items()}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    wire = inline(raw)
    wire["additionalProperties"] = False
    wire["required"] = list(schema.model_fields)
    return wire


def unwrap_envelope(payload: Any, schema: type[BaseModel]) -> Any:
    """Open the packaging a model sometimes puts around the object it was asked for.

    A single-element list, and a one-key object whose key is not a field of the schema, are
    packaging. A one-key object whose key is a field is left alone, so its validation error keeps
    naming what is missing.
    """
    known = set(schema.model_fields)
    for _ in range(3):
        if isinstance(payload, (list, tuple)) and len(payload) == 1:
            payload = payload[0]
            continue
        if isinstance(payload, dict) and len(payload) == 1:
            ((key, inner),) = payload.items()
            if key not in known and isinstance(inner, (dict, list)):
                payload = inner
                continue
        break
    return payload


def _wire_type(node: dict[str, Any]) -> str:
    """How one field is described to a reader that has no schema engine behind it."""
    if "enum" in node:
        return "une valeur parmi " + " | ".join(str(v) for v in node["enum"])
    kind = node.get("type")
    if kind == "number" or kind == "integer":
        low, high = node.get("minimum"), node.get("maximum")
        if low is not None and high is not None:
            return f"nombre entre {low} et {high}"
        return "nombre"
    if kind == "boolean":
        return "true ou false"
    if kind == "string":
        cap = node.get("maxLength")
        return f"chaine de caracteres, {cap} au maximum" if cap else "chaine de caracteres"
    if "const" in node:
        return f"exactement {node['const']}"
    if "anyOf" in node:
        return " ou ".join(_wire_type(branch) for branch in node["anyOf"])
    return kind or "valeur"


def schema_contract(schema: type[BaseModel]) -> str:
    """The output contract as prose, carried in the system message.

    The provider accepts `response_format` in both its `json_schema` and `json_object` forms and
    honours neither, so the shape is imposed by this text instead. It is generated from the
    pydantic model rather than written by hand, and its fingerprint enters the cache key, so
    editing a schema invalidates the answers produced under the previous one. The contract itself
    is sent to the model in French, like the mandates, and is left in that language here.
    """
    wire = strict_json_schema(schema)
    properties: dict[str, Any] = wire.get("properties", {})

    lines = [
        "FORMAT DE SORTIE, IMPERATIF",
        "Rends un unique objet JSON, sans texte avant ni apres, sans bloc de code, sans",
        "commentaire. Exactement les cles suivantes, toutes obligatoires, aucune autre :",
        "",
    ]
    for name, node in properties.items():
        lines.append(f'  "{name}" : {_wire_type(node)}')
        description = str(node.get("description", "")).strip()
        if description:
            for chunk in _wrap(description, 92):
                lines.append(f"      {chunk}")
    lines += [
        "",
        "Aucune cle supplementaire ne sera lue. Une cle manquante invalide la reponse entiere.",
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Deterministic wrap, so the same description hashes identically across versions."""
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def validate_payload(payload: Any, schema: type[BaseModel]) -> tuple[BaseModel | None, str]:
    """Validate, returning the object or the reason it was refused.

    Never substitutes a neutral value for a missing field: a fabricated reading would be
    indistinguishable downstream from a real one and would be counted as a success.
    """
    if payload is None:
        return None, "no JSON object in the response"
    try:
        return schema.model_validate(unwrap_envelope(payload, schema)), ""
    except ValidationError as exc:
        return None, str(exc).replace("\n", " ")[:220]
