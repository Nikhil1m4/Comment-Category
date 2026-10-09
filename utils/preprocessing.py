import re


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_texts(texts):
    return [clean_text(text) for text in texts]