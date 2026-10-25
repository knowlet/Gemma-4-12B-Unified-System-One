"""Action-only DTOs for proposed genuine public-browser supervision.

These objects validate native requests and external annotation consistency.
They do not authenticate observations, guarded execution, ownership, private DOM
proofs, collection completeness or fitting eligibility. Public collection and
fitting integration deliberately remain separate work.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    model_serializer,
    model_validator,
)

from .errors import RequestValidationError
from .jev_generated import TARGETS, _request

PUBLIC_ACTION_SCHEMA = "jev-public-native-action-supervision-v1"
PROOF_FIELDS = frozenset(
    {
        "raw_observation",
        "private_precondition",
        "guarded_transition",
        "private_postcondition",
        "ownership",
        "public_ledger",
    }
)
COMPLETE_ROLES = frozenset({"ordinary_complete", "recovered_complete"})
StateRole = Literal[
    "ordinary_incomplete",
    "correct_document_incomplete",
    "ordinary_complete",
    "recovery_incomplete",
    "recovered_document_incomplete",
    "recovered_complete",
]


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _json_value(value, limit):
    if not isinstance(value, (str, bytes, bytearray)):
        raise ValueError("public JSON must be text or UTF-8 bytes")
    if len(value.encode() if isinstance(value, str) else value) > limit:
        raise ValueError("public JSON exceeds its byte limit")

    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate public JSON key")
            result[key] = item
        return result

    def constant(_):
        raise ValueError("public JSON must contain finite values")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


class NativeWireRequest(RootModel[dict]):
    """Full native request, including ordered heads above the classifier's 52 limit.

    Input and output dictionaries are detached. Nested mutation through ``root``
    is detected before serialization or label access instead of changing the seal.
    No candidate is removed from the stored payload; native ``_request`` performs
    the same 400-target validation used by the generated-action inference path.
    """

    model_config = ConfigDict(strict=True, frozen=True)
    _wire: bytes = PrivateAttr()
    _labels: dict[str, list[str]] = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def detached_payload(cls, value):
        if isinstance(value, cls):
            value._assert_unchanged()
            return copy.deepcopy(value.root)
        if type(value) is not dict:
            raise ValueError("public native request must be an ordinary JSON object")
        return copy.deepcopy(value)

    @model_validator(mode="after")
    def native_contract(self):
        if hasattr(self, "_wire"):
            self._assert_unchanged()
            return self
        identity = self.root.get("model")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("public native request requires its actual model alias")
        try:
            _, source, labels = _request(self.root, identity)
        except RequestValidationError as exc:
            raise ValueError(str(exc)) from exc
        self._wire, self._labels = source, copy.deepcopy(labels)
        return self

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        return cls.model_validate(_json_value(json_data, 1_000_000), **kwargs)

    def _assert_unchanged(self):
        if _encode(self.root) != self._wire:
            raise ValueError("public native request changed after validation")

    @property
    def wire_bytes(self):
        self._assert_unchanged()
        return self._wire

    @property
    def wire_sha256(self):
        return hashlib.sha256(self.wire_bytes).hexdigest()

    @property
    def candidate_labels(self):
        self._assert_unchanged()
        return copy.deepcopy(self._labels)

    @model_serializer(mode="plain")
    def detached_serialization(self):
        self._assert_unchanged()
        return copy.deepcopy(self.root)


class EvidenceReference(BaseModel):
    """A bounded external reference, without a claim that evidence exists."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    artifact_path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"\A[0-9a-f]{64}\z")
    sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def confined_reference(self):
        path = PurePosixPath(self.artifact_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.artifact_path
            or any(character in self.artifact_path for character in "\0\r\n")
            or str(path) != self.artifact_path
        ):
            raise ValueError("public evidence reference must use a confined relative path")
        return self


class PublicNativeActionCase(BaseModel):
    """Strict action annotations; never an EvaluationCase or a DOM proof."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1]
    schema_id: Literal["jev-public-native-action-supervision-v1"]
    id: str = Field(min_length=1, max_length=256)
    split: Literal["train", "development"]
    group_id: str = Field(pattern=r"\Apublic-native-site:[a-z0-9][a-z0-9._-]{0,127}\z")
    task_id: str = Field(min_length=1, max_length=128)
    request: NativeWireRequest
    gold: dict[str, str]
    valid_next_targets: dict[str, list[str]]
    wire_row_id: str = Field(min_length=1, max_length=256)
    episode_id: str = Field(min_length=1, max_length=256)
    pair_id: str = Field(min_length=1, max_length=256)
    state_role: StateRole
    proof_refs: dict[str, EvidenceReference]
    _sealed_fields: bytes = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def detached_metadata(cls, value):
        if not isinstance(value, dict):
            raise ValueError("public action case must be an object")
        if type(value.get("schema_version")) is not int:
            raise ValueError("public action schema_version must be integer1")
        return copy.deepcopy(value)

    @model_validator(mode="after")
    def consistent_action(self):
        if hasattr(self, "_sealed_fields"):
            if self._current_fields() != self._sealed_fields:
                raise ValueError("public action annotations changed after validation")
            return self
        labels = self.request.candidate_labels
        operation = self.gold.get("operation")
        if operation == "BLOCKED":
            raise ValueError("BLOCKED training labels are outside the initial public format")
        if operation not in labels["operation"]:
            raise ValueError("gold operation was not offered by the native request")
        head = TARGETS.get(operation)
        if set(self.gold) != {"operation", *([head] if head else [])}:
            raise ValueError("gold must contain only operation and its active target head")
        if head and self.gold[head] not in labels[head]:
            raise ValueError("gold target was not offered by the native request")
        if not self.valid_next_targets or operation not in self.valid_next_targets:
            raise ValueError("gold action is absent from valid_next_targets")
        for accepted, targets in self.valid_next_targets.items():
            if accepted == "BLOCKED" or accepted not in labels["operation"]:
                raise ValueError("valid_next_targets contains an unsupported offered operation")
            target_head = TARGETS.get(accepted)
            if target_head:
                if (
                    not targets
                    or len(set(targets)) != len(targets)
                    or any(target not in labels[target_head] for target in targets)
                ):
                    raise ValueError("valid_next_targets must contain unique offered target keys")
            elif targets:
                raise ValueError("targetless accepted operations require an empty target list")
        if head and self.gold[head] not in self.valid_next_targets[operation]:
            raise ValueError("gold target is absent from valid_next_targets")
        complete = self.state_role in COMPLETE_ROLES
        if (
            complete != (operation == "DONE")
            or (complete and self.valid_next_targets != {"DONE": []})
            or (not complete and "DONE" in self.valid_next_targets)
        ):
            raise ValueError("DONE annotations require a complete role and no continuing action")
        if set(self.proof_refs) != PROOF_FIELDS:
            raise ValueError("public annotations require exactly the declared external proof refs")
        self._sealed_fields = self._current_fields()
        return self

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        return cls.model_validate(_json_value(json_data, 1_024_000), **kwargs)

    def _current_fields(self):
        fields = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if name == "request":
                value = value.model_dump(mode="json")
            elif name == "proof_refs":
                value = {key: item.model_dump(mode="json") for key, item in value.items()}
            fields[name] = value
        return _encode(fields)

    @model_serializer(mode="wrap")
    def sealed_serialization(self, handler):
        if self._current_fields() != self._sealed_fields:
            raise ValueError("public action annotations changed after validation")
        return handler(self)


def public_training_case(
    payload,
    *,
    case_id,
    group_id,
    task_id,
    split,
    selected,
    valid_targets,
    wire_row_id,
    episode_id,
    pair_id,
    state_role,
    proof_refs,
):
    """Convert one full native wire and external annotations without dropping heads.

    This verifies offered-action consistency only. The caller still needs a
    genuine collector and independently validated raw/guard/private DOM proofs.
    """
    if type(selected) is not dict or set(selected) != {"operation", "target"}:
        raise ValueError("selected must contain exactly operation and target")
    operation, target = selected["operation"], selected["target"]
    if type(operation) is not str:
        raise ValueError("selected operation must be an exact offered string")
    head = TARGETS.get(operation)
    if (head and type(target) is not str) or (not head and target is not None):
        raise ValueError("selected target must be an offered string or null for targetless actions")
    gold = {"operation": operation}
    if head:
        gold[head] = target
    return PublicNativeActionCase(
        schema_version=1,
        schema_id=PUBLIC_ACTION_SCHEMA,
        id=case_id,
        split=split,
        group_id=group_id,
        task_id=task_id,
        request=payload,
        gold=gold,
        valid_next_targets=valid_targets,
        wire_row_id=wire_row_id,
        episode_id=episode_id,
        pair_id=pair_id,
        state_role=state_role,
        proof_refs=proof_refs,
    )
