from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
LABEL_COLORS = {
    "toxicity": "#E4572E",
    "severe_toxicity": "#9B2226",
    "obscene": "#7B2CBF",
    "threat": "#D00000",
    "insult": "#F4A261",
    "identity_attack": "#277DA1",
    "sexual_explicit": "#C9184A",
}

st.set_page_config(
    page_title="Civil Comments Classifier",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource
def load_model_artifact():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.title("💬 Comment Safety Check")
st.write(
    "Check a comment against the Civil Comments categories. Multiple categories "
    "can be flagged for one comment."
)

if not MODEL_PATH.exists():
    st.error("The Civil Comments model has not been trained yet.")
    st.code(
        "python train_civil_comments.py data\\civil_comments\\train.csv",
        language="powershell",
    )
    st.info(
        "Download the Civil Comments training CSV and place it at the path shown "
        "above, or pass its path to the training script."
    )
    st.stop()

artifact = load_model_artifact()
if not isinstance(artifact, dict) or "pipeline" not in artifact or "labels" not in artifact:
    raise ValueError("The saved model artifact is missing its pipeline or label metadata.")

pipeline = artifact["pipeline"]
labels = artifact["labels"]
if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
    raise ValueError("The saved model artifact labels must be a list of strings.")
if not callable(getattr(pipeline, "predict_proba", None)):
    raise ValueError("The saved model pipeline does not support predict_proba.")

model_name = artifact.get(
    "model_name",
    type(pipeline.named_steps.get("classifier", pipeline)).__name__,
)

with st.sidebar:
    st.header("About this model")
    st.write("Dataset: Civil Comments")
    st.write(f"Classifier: TF-IDF + one-vs-rest {model_name}")
    st.write(
        "Training-label cutoff: annotation score >= "
        f"{artifact['label_threshold']:.2f}"
    )
    st.write(f"Fixed detection threshold: {DETECTION_THRESHOLD:.0%}")
    st.caption(
        "Predicted scores are model estimates, not guaranteed probabilities "
        "or moderation decisions."
    )

with st.form("comment_analysis"):
    comment = st.text_area(
        "Comment to analyze",
        height=150,
        placeholder="Type or paste a comment here...",
        help="The saved Civil Comments model scores this text when you submit it.",
    )
    st.caption(
        f"Categories scoring {DETECTION_THRESHOLD:.0%} or higher are flagged."
    )
    submitted = st.form_submit_button(
        "Analyze comment",
        type="primary",
        width="stretch",
    )

if submitted:
    cleaned_comment = clean_text(comment)
    if not cleaned_comment:
        st.warning("Please enter a comment containing text.")
        st.stop()

    probabilities = pipeline.predict_proba([cleaned_comment])[0]
    if len(probabilities) != len(labels):
        raise ValueError(
            "The model output count does not match the label metadata in its artifact."
        )

    result_frame = pd.DataFrame({
        "Label": [LABEL_TITLES.get(label, label.replace("_", " ").title()) for label in labels],
        "Category": labels,
        "Score": probabilities,
    })
    flagged_frame = result_frame.loc[
        result_frame["Score"] >= DETECTION_THRESHOLD
    ].sort_values("Score", ascending=False)

    if flagged_frame.empty:
        st.success(
            f"Neutral — no category reached the {DETECTION_THRESHOLD:.0%} threshold."
        )
        st.caption(
            "No categories crossed this model's flagging threshold. This is an "
            "estimate, not a guarantee that the comment is harmless."
        )
    else:
        flagged_categories = ", ".join(flagged_frame["Label"])
        st.warning(
            f"Potentially harmful content detected in {len(flagged_frame)} "
            f"{'category' if len(flagged_frame) == 1 else 'categories'}: "
            f"{flagged_categories}."
        )
        st.subheader("Flagged categories")
        metric_columns = st.columns(min(3, len(flagged_frame)))
        for index, (_, row) in enumerate(flagged_frame.iterrows()):
            with metric_columns[index % len(metric_columns)]:
                st.metric(label=row["Label"], value=f"{row['Score']:.1%}")

        figure, axis = plt.subplots(figsize=(9, max(2.5, 0.7 * len(flagged_frame))))
        bar_colors = [
            LABEL_COLORS.get(category, "#6C757D")
            for category in flagged_frame["Category"]
        ]
        axis.barh(flagged_frame["Label"], flagged_frame["Score"], color=bar_colors)
        axis.axvline(
            DETECTION_THRESHOLD,
            color="#343A40",
            linestyle="--",
            label=f"Threshold {DETECTION_THRESHOLD:.0%}",
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Estimated score")
        axis.set_title("Flagged Civil Comments categories")
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.grid(axis="x", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    st.caption(
        "Labels are independent, so multiple categories can be flagged. "
        "Scores are model estimates, not guaranteed probabilities or moderation decisions."
    )
