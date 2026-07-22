"""
Task 2 helper: load the best saved model and predict a student's exam score.

Usage examples:
  python prediction.py
  python prediction.py --hours 22 --attendance 90 --previous 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "linear_regression" / "models"
PIPELINE_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
META_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Defaults mirror a typical mid-performing student in the dataset.
DEFAULT_INPUT = {
    "Hours_Studied": 20,
    "Attendance": 80,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "No",
    "Previous_Scores": 70,
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 1,
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "Peer_Influence": "Neutral",
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
}


def load_artifacts():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {PIPELINE_PATH}. "
            "Run the multivariate.ipynb notebook first to train and save the model."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text())
    return pipeline, metadata


def predict_exam_score(features: dict) -> float:
    """Predict exam score from a dict of raw (unscaled) feature values."""
    pipeline, _ = load_artifacts()
    row = {**DEFAULT_INPUT, **features}
    X = pd.DataFrame([row])
    prediction = float(pipeline.predict(X)[0])
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict student exam score with the best Task-1 model."
    )
    parser.add_argument("--hours", type=float, default=DEFAULT_INPUT["Hours_Studied"])
    parser.add_argument("--attendance", type=float, default=DEFAULT_INPUT["Attendance"])
    parser.add_argument("--previous", type=float, default=DEFAULT_INPUT["Previous_Scores"])
    parser.add_argument("--tutoring", type=float, default=DEFAULT_INPUT["Tutoring_Sessions"])
    parser.add_argument(
        "--parental-involvement",
        choices=["Low", "Medium", "High"],
        default=DEFAULT_INPUT["Parental_Involvement"],
    )
    parser.add_argument(
        "--resources",
        choices=["Low", "Medium", "High"],
        default=DEFAULT_INPUT["Access_to_Resources"],
    )
    parser.add_argument(
        "--motivation",
        choices=["Low", "Medium", "High"],
        default=DEFAULT_INPUT["Motivation_Level"],
    )
    parser.add_argument(
        "--family-income",
        choices=["Low", "Medium", "High"],
        default=DEFAULT_INPUT["Family_Income"],
    )
    parser.add_argument(
        "--teacher-quality",
        choices=["Low", "Medium", "High"],
        default=DEFAULT_INPUT["Teacher_Quality"],
    )
    parser.add_argument(
        "--peer-influence",
        choices=["Negative", "Neutral", "Positive"],
        default=DEFAULT_INPUT["Peer_Influence"],
    )
    parser.add_argument(
        "--parental-education",
        choices=["High School", "College", "Postgraduate"],
        default=DEFAULT_INPUT["Parental_Education_Level"],
    )
    parser.add_argument(
        "--distance",
        choices=["Near", "Moderate", "Far"],
        default=DEFAULT_INPUT["Distance_from_Home"],
    )
    parser.add_argument(
        "--extracurricular",
        choices=["Yes", "No"],
        default=DEFAULT_INPUT["Extracurricular_Activities"],
    )
    parser.add_argument(
        "--internet",
        choices=["Yes", "No"],
        default=DEFAULT_INPUT["Internet_Access"],
    )
    parser.add_argument(
        "--learning-disability",
        choices=["Yes", "No"],
        default=DEFAULT_INPUT["Learning_Disabilities"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features = {
        "Hours_Studied": args.hours,
        "Attendance": args.attendance,
        "Previous_Scores": args.previous,
        "Tutoring_Sessions": args.tutoring,
        "Parental_Involvement": args.parental_involvement,
        "Access_to_Resources": args.resources,
        "Motivation_Level": args.motivation,
        "Family_Income": args.family_income,
        "Teacher_Quality": args.teacher_quality,
        "Peer_Influence": args.peer_influence,
        "Parental_Education_Level": args.parental_education,
        "Distance_from_Home": args.distance,
        "Extracurricular_Activities": args.extracurricular,
        "Internet_Access": args.internet,
        "Learning_Disabilities": args.learning_disability,
    }

    pipeline, metadata = load_artifacts()
    X = pd.DataFrame([features])
    score = float(pipeline.predict(X)[0])

    print("Input features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    print()
    if metadata:
        print(f"Model used: {metadata.get('best_model_name', 'unknown')}")
        print(f"Test RMSE (training run): {metadata.get('test_rmse', 'n/a')}")
        print(f"Test R^2 (training run): {metadata.get('test_r2', 'n/a')}")
    print(f"Predicted Exam Score: {score:.2f}")


if __name__ == "__main__":
    main()
