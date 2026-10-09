from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from utils.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "civil_comments_model.pkl"
DETECTION_THRESHOLD = 0.40
LABEL_TITLES = {
    "toxicity": "Toxicity",
    "severe_toxicity": "Severe toxicity",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_attack": "Identity attack",
    "sexual_explicit": "Sexually explicit",
}


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Civil Comments model artifact not found at {model_path}. "
            "This script only runs inference; it does not train a model."
        )

    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict):
        raise ValueError("The model artifact must be a Civil Comments artifact dictionary.")
    if "pipeline" not in artifact or "labels" not in artifact:
        raise ValueError("The model artifact is missing its pipeline or label metadata.")
    if not callable(getattr(artifact["pipeline"], "predict_proba", None)):
        raise ValueError("The artifact pipeline does not support predict_proba.")
    if not isinstance(artifact["labels"], list) or not all(
        isinstance(label, str) for label in artifact["labels"]
    ):
        raise ValueError("The model artifact labels must be a list of strings.")

    return artifact


def analyze_comment(comment: str, artifact: dict) -> None:
    cleaned_comment = clean_text(comment)
    if not cleaned_comment:
        raise ValueError("Please provide a comment containing text.")

    labels = artifact["labels"]
    scores = artifact["pipeline"].predict_proba([cleaned_comment])[0]
    if len(scores) != len(labels):
        raise ValueError(
            "The model output count does not match the label metadata in its artifact."
        )

    print("Label scores:")
    for label, score in zip(labels, scores):
        title = LABEL_TITLES.get(label, label.replace("_", " ").title())
        print(f"  {title}: {score:.1%}")

    detected = [
        LABEL_TITLES.get(label, label.replace("_", " ").title())
        for label, score in zip(labels, scores)
        if score >= DETECTION_THRESHOLD
    ]
    if detected:
        print(
            f"Flagged categories (threshold {DETECTION_THRESHOLD:.0%}): "
            f"{', '.join(detected)}"
        )
    else:
        print(f"Result: Neutral (no category reached {DETECTION_THRESHOLD:.0%}).")

    print(
        "Scores are model estimates, not guaranteed probabilities or moderation decisions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with the existing Civil Comments model. "
            "This script does not fit or retrain a model."
        )
    )
    parser.add_argument("comment", nargs="+", help="Comment text to classify.")
    args = parser.parse_args()

    artifact = load_model_artifact()
    analyze_comment(" ".join(args.comment), artifact)


if __name__ == "__main__":
    main()
