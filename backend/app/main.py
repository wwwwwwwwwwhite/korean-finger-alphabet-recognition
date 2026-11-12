import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .constants import LABELS, MIN_SAMPLES_PER_LABEL
from .model import invalidate_model, load_model, predict, train_model
from .schemas import PredictionInput, SampleInput, SampleUpdate
from .storage import CSV_COLUMNS, DatasetStore, DuplicateSampleError

app = FastAPI(title="Korean Finger Alphabet Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DatasetStore()


def _data_dir() -> Path:
    return store.data_dir


def _not_found_sample() -> HTTPException:
    return HTTPException(status_code=404, detail="Dataset sample not found")


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": load_model(_data_dir()) is not None,
    }


@app.get("/api/labels")
def get_labels() -> dict[str, list[str]]:
    return {"labels": LABELS}


@app.post("/api/samples", status_code=status.HTTP_201_CREATED)
def create_sample(body: SampleInput) -> dict[str, Any]:
    try:
        sample = store.create_sample(body.label, body.landmarks, body.session_id)
    except DuplicateSampleError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_sample",
                "message": "A near-identical sample with this label already exists.",
                "existing_sample_id": exc.sample_id,
            },
        ) from exc
    invalidate_model(_data_dir())
    return sample


@app.get("/api/samples")
def list_samples(label: Optional[str] = Query(default=None)) -> list[dict[str, str]]:
    if label is not None and label not in LABELS:
        raise HTTPException(status_code=422, detail="Unsupported label")
    samples = store.list_samples()
    if label is not None:
        samples = [sample for sample in samples if sample["label"] == label]
    return [{"id": sample["id"], "label": sample["label"]} for sample in samples]


@app.get("/api/samples/summary")
def dataset_summary() -> dict[str, Any]:
    samples = store.list_samples()
    counts = dict.fromkeys(LABELS, 0)
    for sample in samples:
        counts[sample["label"]] += 1
    return {"total": len(samples), "by_label": counts}


@app.get("/api/samples/export")
def export_samples() -> Response:
    return export_dataset()


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str) -> dict[str, Any]:
    sample = store.get_sample(sample_id)
    if sample is None:
        raise _not_found_sample()
    return sample


@app.put("/api/samples/{sample_id}")
def update_sample(sample_id: str, body: SampleUpdate) -> dict[str, Any]:
    try:
        updated = store.update_sample(
            sample_id,
            label=body.label,
            landmarks=body.landmarks,
        )
    except DuplicateSampleError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_sample",
                "message": "A near-identical sample with this label already exists.",
                "existing_sample_id": exc.sample_id,
            },
        ) from exc
    if updated is None:
        raise _not_found_sample()
    invalidate_model(_data_dir())
    return updated


@app.delete("/api/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(sample_id: str) -> Response:
    if not store.delete_sample(sample_id):
        raise _not_found_sample()
    invalidate_model(_data_dir())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    samples = store.list_samples()
    counts = dict.fromkeys(LABELS, 0)
    for sample in samples:
        counts[sample["label"]] += 1
    return {
        "total_samples": len(samples),
        "categories": counts,
        "minimum_samples_per_label_for_training": MIN_SAMPLES_PER_LABEL,
    }


@app.get("/api/dataset/export.csv")
def export_dataset() -> Response:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for sample in store.list_samples():
        writer.writerow(
            {
                **sample,
                "landmarks": json.dumps(sample["landmarks"], ensure_ascii=False),
            }
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="dataset.csv"'},
    )


@app.post("/api/train")
def train() -> dict[str, Any]:
    try:
        return train_model(store.list_samples(), _data_dir())
    except ValueError as exc:
        if exc.args and isinstance(exc.args[0], dict):
            raise HTTPException(status_code=409, detail=exc.args[0]) from exc
        raise


@app.get("/api/model/report")
def model_report() -> dict[str, Any]:
    try:
        artifact = load_model(_data_dir())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if artifact is None:
        raise HTTPException(status_code=404, detail="Model has not been trained")
    return artifact["report"]


@app.post("/api/predict")
def predict_label(body: PredictionInput) -> dict[str, Any]:
    try:
        artifact = load_model(_data_dir())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if artifact is None:
        raise HTTPException(status_code=404, detail="Model has not been trained")
    return predict(artifact, body.landmarks)
