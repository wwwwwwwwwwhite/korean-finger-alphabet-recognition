from math import cos, sin
from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from app import main
from app.constants import LABELS
from app.storage import DatasetStore


@pytest.fixture
def client(monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / ".test-data"
    shutil.rmtree(data_dir, ignore_errors=True)
    monkeypatch.setattr(main, "store", DatasetStore(data_dir))
    try:
        yield TestClient(main.app)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def landmarks(seed=0):
    return [
        {
            "x": sin(index * 0.21 + seed * 0.13),
            "y": cos(index * 0.31 + seed * 0.17),
            "z": sin(index * 0.11 + seed * 0.23),
        }
        for index in range(21)
    ]


def test_dataset_crud_and_export(client):
    created = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": landmarks()},
    )
    assert created.status_code == 201
    sample = created.json()

    listed = client.get("/api/samples", params={"label": "ㄱ"})
    assert listed.status_code == 200
    assert listed.json() == [{"id": sample["id"], "label": "ㄱ"}]
    assert client.get("/api/samples/summary").json()["total"] == 1
    assert client.get("/api/samples/summary").json()["by_label"]["ㄱ"] == 1

    updated = client.put(
        f"/api/samples/{sample['id']}",
        json={"label": "ㄴ"},
    )
    assert updated.status_code == 200
    assert updated.json()["label"] == "ㄴ"

    exported = client.get("/api/samples/export")
    assert exported.status_code == 200
    assert "landmarks" in exported.text
    assert "ㄴ" in exported.text

    deleted = client.delete(f"/api/samples/{sample['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/samples/{sample['id']}").status_code == 404


def test_rejects_non_21_landmarks(client):
    response = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": landmarks()[:20]},
    )
    assert response.status_code == 422


def test_accepts_xyz_triplets(client):
    points = [[index / 100, index / 200, index / 300] for index in range(21)]
    response = client.post(
        "/api/samples",
        json={
            "label": "ㄱ",
            "landmarks": points,
        },
    )
    assert response.status_code == 201
    assert response.json()["landmarks"][0] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_persists_optional_session_id(client):
    session_id = "capture-run-2026-11-12-a"
    response = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": landmarks(11), "session_id": session_id},
    )
    assert response.status_code == 201
    assert response.json()["session_id"] == session_id
    persisted_sample = DatasetStore(main.store.data_dir).list_samples()[0]
    assert persisted_sample["session_id"] == session_id

    legacy_sample = client.post(
        "/api/samples",
        json={"label": "ㄴ", "landmarks": landmarks(13)},
    )
    assert legacy_sample.status_code == 201
    assert legacy_sample.json()["session_id"] is None


def test_rejects_exact_and_near_duplicate_same_label(client):
    original = landmarks(1)
    created = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": original},
    )
    assert created.status_code == 201

    exact_duplicate = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": original},
    )
    assert exact_duplicate.status_code == 409
    assert exact_duplicate.json()["detail"]["code"] == "duplicate_sample"

    slightly_noisy = [point.copy() for point in original]
    slightly_noisy[0]["x"] += 0.001
    near_duplicate = client.post(
        "/api/samples",
        json={"label": "ㄱ", "landmarks": slightly_noisy},
    )
    assert near_duplicate.status_code == 409

    other_label_same_landmarks = client.post(
        "/api/samples",
        json={"label": "ㄴ", "landmarks": original},
    )
    assert other_label_same_landmarks.status_code == 201


def test_ignores_optional_browser_landmark_metadata(client):
    points = [
        {"x": index / 100, "y": index / 200, "z": index / 300, "visibility": 0.9}
        for index in range(21)
    ]
    response = client.post("/api/samples", json={"label": "ㄱ", "landmarks": points})
    assert response.status_code == 201
    assert set(response.json()["landmarks"][0]) == {"x", "y", "z"}


def test_training_requires_all_labels_and_prediction_requires_model(client):
    assert client.get("/api/health").json() == {
        "status": "ok",
        "model_ready": False,
    }
    for seed in (0, 5):
        response = client.post(
            "/api/samples",
            json={"label": "ㄱ", "landmarks": landmarks(seed)},
        )
        assert response.status_code == 201
    train_response = client.post("/api/train")
    assert train_response.status_code == 409
    assert train_response.json()["detail"]["code"] == "insufficient_categories"
    assert train_response.json()["detail"]["categories"] == {"ㄱ": 2}
    assert train_response.json()["detail"]["required_per_label"] == 3

    prediction = client.post("/api/predict", json={"landmarks": landmarks()})
    assert prediction.status_code == 404
    assert client.get("/api/model/report").status_code == 404


def test_custom_labels_can_be_saved_and_trained(client):
    created_label = client.post("/api/labels", json={"label": "你好"})
    assert created_label.status_code == 201
    assert "你好" in created_label.json()["labels"]
    assert "你好" in DatasetStore(main.store.data_dir).list_labels()
    assert client.post("/api/labels", json={"label": "你好"}).status_code == 409
    assert client.post(
        "/api/samples",
        json={"label": "未登记", "landmarks": landmarks(50)},
    ).status_code == 422

    for label, offset in (("ㄱ", 0), ("你好", 10)):
        for variation in range(3):
            response = client.post(
                "/api/samples",
                json={
                    "label": label,
                    "landmarks": landmarks(offset + variation),
                },
            )
            assert response.status_code == 201

    assert client.get("/api/samples/summary").json()["by_label"]["你好"] == 3
    report_response = client.post("/api/train")
    assert report_response.status_code == 200
    assert report_response.json()["labels"] == ["ㄱ", "你好"]
    assert client.get("/api/health").json()["model_ready"] is True
    assert (main.store.data_dir / "random_forest.joblib").exists()

    deleted_label = client.delete("/api/labels", params={"label": "你好"})
    assert deleted_label.status_code == 200
    assert deleted_label.json()["deleted_samples"] == 3
    assert "你好" not in deleted_label.json()["labels"]
    assert client.get("/api/samples/summary").json()["total"] == 3
    assert client.get("/api/samples", params={"label": "你好"}).status_code == 422
    assert client.get("/api/health").json()["model_ready"] is False
    assert not (main.store.data_dir / "random_forest.joblib").exists()

    assert client.delete("/api/labels", params={"label": "ㄱ"}).status_code == 409
    assert client.delete("/api/labels", params={"label": "不存在"}).status_code == 404


def test_training_report_and_prediction(client):
    for label_index, label in enumerate(LABELS):
        for variation in range(3):
            response = client.post(
                "/api/samples",
                json={
                    "label": label,
                    "landmarks": landmarks(label_index * 3 + variation),
                },
            )
            assert response.status_code == 201

    report_response = client.post("/api/train")
    assert report_response.status_code == 200
    report = report_response.json()
    assert {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "labels",
        "per_class",
        "confusion_matrix",
        "train_size",
        "test_size",
    }.issubset(report)
    assert report["train_size"] + report["test_size"] == 3 * len(LABELS)
    assert "Capture sessions are not tracked" in report["metric_caveat"]
    assert all(
        0.0 <= report[metric] <= 1.0
        for metric in ("accuracy", "precision", "recall", "f1_score")
    )
    assert len(report["per_class"]) == len(LABELS)
    assert len(report["confusion_matrix"]) == len(LABELS)
    assert client.get("/api/health").json()["model_ready"] is True

    prediction = client.post("/api/predict", json={"landmarks": landmarks(0)})
    assert prediction.status_code == 200
    assert set(prediction.json()) == {"label", "confidence"}
    assert prediction.json()["label"] in LABELS
    assert 0.0 <= prediction.json()["confidence"] <= 1.0
    assert client.get("/api/model/report").json() == report

    added_sample = client.post(
        "/api/samples",
        json={"label": LABELS[0], "landmarks": landmarks(100)},
    )
    assert added_sample.status_code == 201
    assert client.get("/api/model/report").status_code == 404
    assert client.post("/api/predict", json={"landmarks": landmarks(0)}).status_code == 404
