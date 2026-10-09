from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "civil_comments_logistic_model.pkl"
LABEL_COLUMNS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def load_civil_comments(csv_path: Path, label_threshold: float) -> tuple[pd.Series, pd.DataFrame]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Civil Comments CSV not found at {csv_path}. "
            "Download the dataset and pass the path to its CSV file."
        )

    columns = pd.read_csv(csv_path, nrows=0).columns
    text_column = next(
        (column for column in ("text", "comment_text") if column in columns),
        None,
    )
    if text_column is None:
        raise ValueError("Expected a Civil Comments text column named 'text' or 'comment_text'.")

    available_labels = set(columns)
    source_columns = {
        "toxicity": "toxicity" if "toxicity" in available_labels else "target",
        **{label: label for label in LABEL_COLUMNS if label != "toxicity"},
    }
    missing_labels = [
        label for label, column in source_columns.items()
        if column not in available_labels
    ]
    if missing_labels:
        raise ValueError(
            "This CSV does not contain the seven supported Civil Comments label columns: "
            + ", ".join(missing_labels)
        )

    required_source_labels = list(dict.fromkeys(source_columns.values()))
    frame = pd.read_csv(csv_path, usecols=[text_column, *required_source_labels])
    frame = frame.rename(columns={
        source_column: label
        for label, source_column in source_columns.items()
        if source_column != label
    })
    frame = frame.dropna(subset=LABEL_COLUMNS)
    texts = frame[text_column].fillna("").astype(str).map(clean_text)
    target_scores = frame[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_rows = target_scores.notna().all(axis=1) & texts.ne("")
    texts = texts.loc[valid_rows].reset_index(drop=True)
    target_scores = target_scores.loc[valid_rows].reset_index(drop=True)

    if target_scores.empty:
        raise ValueError("No rows with usable text and all required label scores were found.")
    if ((target_scores < 0) | (target_scores > 1)).any().any():
        raise ValueError("Civil Comments target annotation scores must be between 0 and 1.")

    targets = target_scores.ge(label_threshold).astype("uint8")
    absent_labels = [
        label for label in LABEL_COLUMNS
        if targets[label].nunique() < 2
    ]
    if absent_labels:
        raise ValueError(
            "These labels do not have both positive and negative examples after "
            f"applying threshold {label_threshold:.2f}: {', '.join(absent_labels)}. "
            "Use a larger or more representative data file."
        )

    return texts, targets


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=250_000,
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multi-label model on a Civil Comments CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "civil_comments" / "train.csv"),
        help="Civil Comments CSV containing text/comment_text and seven label score columns.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Convert annotation scores >= this value to positive labels (default: 0.5).",
    )
    parser.add_argument(
        "--model-output",
        default=str(MODEL_PATH),
        help="Output path for the trained model artifact.",
    )
    args = parser.parse_args()

    if not 0 < args.label_threshold < 1:
        raise ValueError("--label-threshold must be greater than 0 and less than 1.")

    texts, targets = load_civil_comments(Path(args.csv_path), args.label_threshold)
    print(f"Usable rows: {len(texts):,}")
    print("Positive label counts after thresholding:")
    for label in LABEL_COLUMNS:
        print(f"  {label}: {int(targets[label].sum()):,} ({targets[label].mean():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        targets,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\nHeld-out test classification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=LABEL_COLUMNS,
        digits=4,
        zero_division=0,
    ))
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(
        "This is a random holdout score. For final performance reporting, keep a "
        "separate untouched dataset split and document the split methodology."
    )

    pipeline.fit(texts, targets)
    artifact = {
        "pipeline": pipeline,
        "labels": LABEL_COLUMNS,
        "label_threshold": args.label_threshold,
        "dataset": "Civil Comments",
    }
    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)
    print(f"\nSaved multi-label model artifact to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
