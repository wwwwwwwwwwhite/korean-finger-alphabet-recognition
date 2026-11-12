import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .constants import LABELS, MIN_SAMPLES_PER_LABEL
from .normalization import normalize_landmarks
from .schemas import Landmark


def model_path(data_dir: Path) -> Path:
    return data_dir / "random_forest.joblib"


def invalidate_model(data_dir: Path) -> None:
    model_path(data_dir).unlink(missing_ok=True)


def category_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(LABELS, 0)
    for sample in samples:
        if sample["label"] in counts:
            counts[sample["label"]] += 1
    return counts


def train_model(samples: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    counts = category_counts(samples)
    insufficient = {
        label: count for label, count in counts.items() if count < MIN_SAMPLES_PER_LABEL
    }
    if insufficient:
        raise ValueError(
            {
                "code": "insufficient_categories",
                "message": (
                    "Training requires at least three recorded samples for every "
                    "supported label: two for training and one for holdout evaluation."
                ),
                "required_per_label": MIN_SAMPLES_PER_LABEL,
                "categories": insufficient,
            }
        )

    features = np.vstack(
        [
            normalize_landmarks(
                [Landmark.model_validate(point) for point in sample["landmarks"]]
            )
            for sample in samples
        ]
    )
    targets = np.asarray([sample["label"] for sample in samples])
    estimator = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    test_size = max(len(LABELS), int(np.ceil(len(samples) * 0.2)))
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=42,
        stratify=targets,
    )
    estimator.fit(train_features, train_targets)
    test_predictions = estimator.predict(test_features)
    report_data = classification_report(
        test_targets,
        test_predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(test_targets, test_predictions, labels=LABELS)
    per_class = [
        {
            "label": label,
            "precision": float(report_data[label]["precision"]),
            "recall": float(report_data[label]["recall"]),
            "f1_score": float(report_data[label]["f1-score"]),
            "support": int(report_data[label]["support"]),
        }
        for label in LABELS
    ]
    report = {
        "labels": LABELS,
        "accuracy": float(accuracy_score(test_targets, test_predictions)),
        "precision": float(report_data["macro avg"]["precision"]),
        "recall": float(report_data["macro avg"]["recall"]),
        "f1_score": float(report_data["macro avg"]["f1-score"]),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "train_size": len(train_targets),
        "test_size": len(test_targets),
        "metric_caveat": (
            "Metrics use a stratified sample-level holdout after rejecting near-identical "
            "same-label landmarks. Capture sessions are not tracked, so correlated samples "
            "from separate captures may still make these metrics optimistic."
        ),
    }

    estimator.fit(features, targets)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = model_path(data_dir)
    temporary_path = destination.with_suffix(".joblib.tmp")
    joblib.dump({"estimator": estimator, "report": report}, temporary_path)
    os.replace(temporary_path, destination)
    return report


def load_model(data_dir: Path) -> Optional[dict[str, Any]]:
    path = model_path(data_dir)
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "estimator" not in artifact or "report" not in artifact:
        raise ValueError("saved model artifact is invalid")
    return artifact


def predict(artifact: dict[str, Any], landmarks: list[Landmark]) -> dict[str, Any]:
    features = normalize_landmarks(landmarks).reshape(1, -1)
    estimator = artifact["estimator"]
    probabilities = estimator.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    return {
        "label": str(estimator.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
    }
