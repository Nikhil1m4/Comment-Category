# Civil Comments Classifier

Streamlit app that scores comments against seven independent Civil Comments
categories. A single comment can receive more than one flagged category.

## Run locally

Requirements: Python 3.12 and the model file `civil_comments_model.pkl`.

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app applies a fixed inference threshold of 40%. If no category reaches
that score, it reports Neutral; otherwise, it shows the flagged categories and
their scores. These are model estimates, not guaranteed probabilities or
moderation decisions.

## Deploy on Streamlit Community Cloud

Push this repository to GitHub, then create a Community Cloud app using
`app.py` as the main file. The included `civil_comments_model.pkl` is required
by the app and is intentionally included in Git; it is about 10 MB. The
training CSV and local credentials are not included.

## Model and optional training

The included serving artifact is a TF-IDF + LightGBM multi-label model trained
on Civil Comments. The separate `train_civil_comments.py` script can train a
TF-IDF + one-vs-rest Logistic Regression artifact from a Civil Comments CSV
with a `text` or `comment_text` column and the seven target columns.

```powershell
python train_civil_comments.py data\civil_comments\train.csv
```

The training script converts annotation scores at or above `0.5` to positive
labels by default, reports a random holdout evaluation, then refits on all
provided rows and saves a separate `civil_comments_logistic_model.pkl` file.
It will not overwrite the LightGBM artifact used by the deployed app. Training
is not needed to run or deploy the included serving model.

Only load model pickle files from trusted sources. The included artifact is
required for app deployment and is tracked by Git.

Civil Comments dataset sources:

- [Civil Comments in TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/civil_comments)
- [Civil Comments / Jigsaw Unintended Bias data on Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

## Running inference from the command line

`tune_models.py` loads the same saved model as the app. It does not train or
modify the model:

```powershell
python tune_models.py "I love you"
```

## Files

- `app.py` — Streamlit interface.
- `civil_comments_model.pkl` — pre-trained model artifact required by the app.
- `train_civil_comments.py` — optional model training script; writes a separate artifact.
- `tune_models.py` — command-line inference.
- `utils/preprocessing.py` — text cleaning shared by training and inference.
- `requirements.txt` — Python dependencies.
