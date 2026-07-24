"""
Task 2: FastAPI service for student exam-score prediction and model retraining.

Run locally:
  uvicorn summative.API.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "summative" / "linear_regression" / "models"
DATA_PATH = ROOT / "summative" / "data" / "StudentPerformanceFactors.csv"
PIPELINE_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
META_PATH = ARTIFACTS_DIR / "model_metadata.json"

TARGET = "Exam_Score"

NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
]

ORDINAL_MAPS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}

BINARY_FEATURES = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + list(ORDINAL_MAPS.keys()) + BINARY_FEATURES

_model_lock = threading.Lock()
_pipeline: Optional[Pipeline] = None
_metadata: dict = {}


class LowMedHigh(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class PeerInfluence(str, Enum):
    negative = "Negative"
    neutral = "Neutral"
    positive = "Positive"


class ParentalEducation(str, Enum):
    high_school = "High School"
    college = "College"
    postgraduate = "Postgraduate"


class DistanceFromHome(str, Enum):
    near = "Near"
    moderate = "Moderate"
    far = "Far"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class StudentFeatures(BaseModel):
    """Input features required by the linear regression pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Hours_Studied": 20,
                "Attendance": 85,
                "Previous_Scores": 75,
                "Tutoring_Sessions": 1,
                "Parental_Involvement": "Medium",
                "Access_to_Resources": "Medium",
                "Motivation_Level": "Medium",
                "Family_Income": "Medium",
                "Teacher_Quality": "Medium",
                "Peer_Influence": "Neutral",
                "Parental_Education_Level": "College",
                "Distance_from_Home": "Near",
                "Extracurricular_Activities": "No",
                "Internet_Access": "Yes",
                "Learning_Disabilities": "No",
            }
        }
    )

    Hours_Studied: Annotated[float, Field(ge=0, le=60, description="Weekly study hours (0–60)")]
    Attendance: Annotated[float, Field(ge=0, le=100, description="Class attendance percentage (0–100)")]
    Previous_Scores: Annotated[float, Field(ge=0, le=100, description="Prior exam score (0–100)")]
    Tutoring_Sessions: Annotated[
        int, Field(ge=0, le=20, description="Number of tutoring sessions (0–20)")
    ]

    Parental_Involvement: LowMedHigh
    Access_to_Resources: LowMedHigh
    Motivation_Level: LowMedHigh
    Family_Income: LowMedHigh
    Teacher_Quality: LowMedHigh
    Peer_Influence: PeerInfluence
    Parental_Education_Level: ParentalEducation
    Distance_from_Home: DistanceFromHome

    Extracurricular_Activities: YesNo
    Internet_Access: YesNo
    Learning_Disabilities: YesNo


class PredictionResponse(BaseModel):
    predicted_exam_score: float
    model_name: str
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None


class RetrainRow(StudentFeatures):
    """One labelled row used to retrain the model (features + target)."""

    Exam_Score: Annotated[float, Field(ge=0, le=110, description="Observed exam score")]


class RetrainRequest(BaseModel):
    """JSON body for streaming / uploading new labelled samples."""

    samples: Annotated[List[RetrainRow], Field(min_length=1)]
    append_to_existing: bool = Field(
        default=True,
        description="If true, combine new samples with the original CSV before refitting.",
    )


class RetrainResponse(BaseModel):
    status: str
    n_new_samples: int
    n_total_samples: int
    test_rmse: float
    test_mae: float
    test_r2: float
    model_path: str
    retrained_at: str


def _load_pipeline() -> tuple[Pipeline, dict]:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {PIPELINE_PATH}. "
            "Run multivariate.ipynb first to train and save the model."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    metadata: dict = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text())
    return pipeline, metadata


def get_pipeline() -> Pipeline:
    global _pipeline, _metadata
    with _model_lock:
        if _pipeline is None:
            _pipeline, _metadata = _load_pipeline()
        return _pipeline


def refresh_pipeline() -> None:
    """Force-reload artifacts from disk (called after retrain)."""
    global _pipeline, _metadata
    with _model_lock:
        _pipeline, _metadata = _load_pipeline()


def build_fresh_pipeline() -> Pipeline:
    """Rebuild the same Ridge + preprocessing pipeline used in Task 1."""
    ordinal_features = list(ORDINAL_MAPS.keys())
    ordinal_categories = [ORDINAL_MAPS[c] for c in ordinal_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "ord",
                Pipeline(
                    [
                        ("encoder", OrdinalEncoder(categories=ordinal_categories)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ordinal_features,
            ),
            (
                "bin",
                Pipeline(
                    [
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[["No", "Yes"]] * len(BINARY_FEATURES)
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                BINARY_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", Ridge(random_state=42)),
        ]
    )


def _dataframe_from_features(features: StudentFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])


def _validate_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded data is missing required columns: {missing}",
        )
    frame = df[required].dropna().copy()
    if frame.empty:
        raise HTTPException(status_code=400, detail="No usable rows after dropping NaNs.")
    return frame


def retrain_model(new_df: pd.DataFrame, append_to_existing: bool = True) -> RetrainResponse:
    new_clean = _validate_training_frame(new_df)

    if append_to_existing:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Base dataset not found at {DATA_PATH}",
            )
        base = pd.read_csv(DATA_PATH)
        base_clean = _validate_training_frame(base)
        combined = pd.concat([base_clean, new_clean], ignore_index=True)
    else:
        combined = new_clean

    if len(combined) < 10:
        raise HTTPException(
            status_code=400,
            detail="Need at least 10 labelled rows to retrain reliably.",
        )

    X = combined[FEATURE_COLUMNS]
    y = combined[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_fresh_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with _model_lock:
        joblib.dump(pipeline, PIPELINE_PATH)
        metadata = {}
        if META_PATH.exists():
            metadata = json.loads(META_PATH.read_text())
        metadata.update(
            {
                "best_model_name": "Ridge Regression (retrained)",
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                "retrained_at": datetime.now(timezone.utc).isoformat(),
                "n_training_rows": int(len(combined)),
            }
        )
        META_PATH.write_text(json.dumps(metadata, indent=2))
        global _pipeline, _metadata
        _pipeline = pipeline
        _metadata = metadata

    return RetrainResponse(
        status="retrained",
        n_new_samples=int(len(new_clean)),
        n_total_samples=int(len(combined)),
        test_rmse=rmse,
        test_mae=mae,
        test_r2=r2,
        model_path=str(PIPELINE_PATH),
        retrained_at=metadata["retrained_at"],
    )


app = FastAPI(
    title="Student Exam Score Predictor",
    description=(
        "Serves the Task-1 Ridge regression pipeline for predicting exam scores "
        "from study/support factors, and exposes a retraining endpoint for new labelled data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_load_model() -> None:
    try:
        refresh_pipeline()
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")


@app.get("/", tags=["health"])
def root():
    return {
        "message": "Student Exam Score API",
        "docs": "/docs",
        "predict": "POST /predict",
        "retrain": "POST /retrain",
        "retrain_csv": "POST /retrain/csv",
    }


@app.get("/health", tags=["health"])
def health():
    model_loaded = _pipeline is not None or PIPELINE_PATH.exists()
    return {"status": "ok" if model_loaded else "model_missing", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: StudentFeatures):
    """Predict exam score from student study/support factors."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    X = _dataframe_from_features(features)
    with _model_lock:
        score = float(pipeline.predict(X)[0])
        meta = dict(_metadata)

    return PredictionResponse(
        predicted_exam_score=round(score, 2),
        model_name=meta.get("best_model_name", "unknown"),
        test_rmse=meta.get("test_rmse"),
        test_r2=meta.get("test_r2"),
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["retraining"])
def retrain(body: RetrainRequest):
    """Retrain the Ridge pipeline using newly uploaded/streamed labelled samples."""
    rows = [sample.model_dump() for sample in body.samples]
    new_df = pd.DataFrame(rows)
    return retrain_model(new_df, append_to_existing=body.append_to_existing)


@app.post("/retrain/csv", response_model=RetrainResponse, tags=["retraining"])
async def retrain_csv(
    file: UploadFile = File(..., description="CSV with feature columns + Exam_Score"),
    append_to_existing: bool = True,
):
    """Retrain from an uploaded CSV file of labelled student records."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    return retrain_model(new_df, append_to_existing=append_to_existing)
