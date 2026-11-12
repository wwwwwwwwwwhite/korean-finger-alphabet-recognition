import csv
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .constants import LABELS, MAX_NORMALIZED_LANDMARK_RMS_DISTANCE
from .normalization import normalize_landmarks
from .schemas import Landmark, normalize_label

CSV_COLUMNS = ("id", "label", "landmarks", "session_id", "created_at", "updated_at")


class DuplicateSampleError(ValueError):
    def __init__(self, sample_id: str) -> None:
        super().__init__(sample_id)
        self.sample_id = sample_id


def default_data_dir() -> Path:
    configured = os.environ.get("FINGER_ALPHABET_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


class DatasetStore:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.csv_path = self.data_dir / "dataset.csv"
        self.custom_labels_path = self.data_dir / "custom_labels.json"
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _deserialize_row(self, row: dict[str, str]) -> dict[str, Any]:
        try:
            label = normalize_label(row["label"])
            raw_points = json.loads(row["landmarks"])
            points = [Landmark.model_validate(point).model_dump() for point in raw_points]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("dataset CSV contains an invalid landmark row") from exc
        if len(points) != 21:
            raise ValueError("dataset CSV contains an invalid landmark count")
        return {
            "id": row["id"],
            "label": label,
            "landmarks": points,
            "session_id": row.get("session_id") or None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as source:
            return [self._deserialize_row(row) for row in csv.DictReader(source)]

    def _write_unlocked(self, rows: list[dict[str, Any]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "session_id": row.get("session_id") or "",
                        "landmarks": json.dumps(
                            row["landmarks"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )

    def list_samples(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read_unlocked()

    def list_labels(self) -> list[str]:
        with self._lock:
            custom_labels: list[str] = []
            if self.custom_labels_path.exists():
                with self.custom_labels_path.open("r", encoding="utf-8") as source:
                    stored = json.load(source)
                if not isinstance(stored, list) or not all(isinstance(item, str) for item in stored):
                    raise ValueError("custom label registry is invalid")
                custom_labels = [normalize_label(item) for item in stored]
            return [*LABELS, *(label for label in custom_labels if label not in LABELS)]

    def add_custom_label(self, label: str) -> list[str]:
        label = normalize_label(label)
        with self._lock:
            labels = self.list_labels()
            if label in labels:
                raise ValueError("label already exists")
            custom_labels = [item for item in labels if item not in LABELS]
            custom_labels.append(label)
            self._write_custom_labels_unlocked(custom_labels)
            return [*LABELS, *custom_labels]

    def delete_custom_label(self, label: str) -> tuple[list[str], int]:
        label = normalize_label(label)
        with self._lock:
            labels = self.list_labels()
            if label in LABELS:
                raise ValueError("built-in labels cannot be deleted")
            if label not in labels:
                raise KeyError(label)

            rows = self._read_unlocked()
            remaining_rows = [row for row in rows if row["label"] != label]
            deleted_samples = len(rows) - len(remaining_rows)
            if deleted_samples:
                self._write_unlocked(remaining_rows)

            custom_labels = [item for item in labels if item not in LABELS and item != label]
            self._write_custom_labels_unlocked(custom_labels)
            return [*LABELS, *custom_labels], deleted_samples

    def _write_custom_labels_unlocked(self, labels: list[str]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.custom_labels_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as destination:
            json.dump(labels, destination, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self.custom_labels_path)

    def get_sample(self, sample_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return next(
                (sample for sample in self._read_unlocked() if sample["id"] == sample_id),
                None,
            )

    def create_sample(
        self,
        label: str,
        landmarks: list[Landmark],
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            rows = self._read_unlocked()
            duplicate_id = self._find_duplicate(label, landmarks, rows)
            if duplicate_id is not None:
                raise DuplicateSampleError(duplicate_id)
            now = self._now()
            sample = {
                "id": str(uuid.uuid4()),
                "label": label,
                "landmarks": [point.model_dump() for point in landmarks],
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
            }
            rows.append(sample)
            self._write_unlocked(rows)
            return sample

    def update_sample(
        self,
        sample_id: str,
        *,
        label: Optional[str] = None,
        landmarks: Optional[list[Landmark]] = None,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            rows = self._read_unlocked()
            sample = next((row for row in rows if row["id"] == sample_id), None)
            if sample is None:
                return None
            updated_label = label if label is not None else sample["label"]
            updated_landmarks = (
                [point.model_dump() for point in landmarks]
                if landmarks is not None
                else sample["landmarks"]
            )
            points = [Landmark.model_validate(point) for point in updated_landmarks]
            duplicate_id = self._find_duplicate(
                updated_label,
                points,
                rows,
                exclude_id=sample_id,
            )
            if duplicate_id is not None:
                raise DuplicateSampleError(duplicate_id)
            sample["label"] = updated_label
            sample["landmarks"] = updated_landmarks
            sample["updated_at"] = self._now()
            self._write_unlocked(rows)
            return sample

    @staticmethod
    def _find_duplicate(
        label: str,
        landmarks: list[Landmark],
        rows: list[dict[str, Any]],
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        candidate = normalize_landmarks(landmarks).reshape(21, 3)
        for row in rows:
            if row["label"] != label or row["id"] == exclude_id:
                continue
            existing = normalize_landmarks(
                [Landmark.model_validate(point) for point in row["landmarks"]]
            ).reshape(21, 3)
            rms_distance = float(np.sqrt(np.mean(np.sum((candidate - existing) ** 2, axis=1))))
            if rms_distance <= MAX_NORMALIZED_LANDMARK_RMS_DISTANCE:
                return row["id"]
        return None

    def delete_sample(self, sample_id: str) -> bool:
        with self._lock:
            rows = self._read_unlocked()
            remaining = [row for row in rows if row["id"] != sample_id]
            if len(remaining) == len(rows):
                return False
            self._write_unlocked(remaining)
            return True
