from math import isfinite
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .constants import LABELS, LANDMARK_COUNT


class Landmark(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x: float
    y: float
    z: float

    @model_validator(mode="before")
    @classmethod
    def accept_coordinate_triplets(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return {"x": value[0], "y": value[1], "z": value[2]}
        return value

    @field_validator("x", "y", "z")
    @classmethod
    def require_finite_coordinate(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("landmark coordinates must be finite numbers")
        return value


class SampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    landmarks: list[Landmark]
    session_id: Optional[str] = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError(f"label must be one of the supported labels: {', '.join(LABELS)}")
        return value

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(cls, value: list[Landmark]) -> list[Landmark]:
        if len(value) != LANDMARK_COUNT:
            raise ValueError(f"exactly {LANDMARK_COUNT} landmarks are required")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 128:
            raise ValueError("session_id must contain 1 to 128 characters")
        return value


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    landmarks: list[Landmark]

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(cls, value: list[Landmark]) -> list[Landmark]:
        if len(value) != LANDMARK_COUNT:
            raise ValueError(f"exactly {LANDMARK_COUNT} landmarks are required")
        return value


class SampleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Optional[str] = None
    landmarks: Optional[list[Landmark]] = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in LABELS:
            raise ValueError(f"label must be one of the supported labels: {', '.join(LABELS)}")
        return value

    @field_validator("landmarks")
    @classmethod
    def validate_landmark_count(
        cls, value: Optional[list[Landmark]]
    ) -> Optional[list[Landmark]]:
        if value is not None and len(value) != LANDMARK_COUNT:
            raise ValueError(f"exactly {LANDMARK_COUNT} landmarks are required")
        return value

    @model_validator(mode="after")
    def require_update_fields(self) -> "SampleUpdate":
        if self.label is None and self.landmarks is None:
            raise ValueError("at least one of label or landmarks must be provided")
        return self
