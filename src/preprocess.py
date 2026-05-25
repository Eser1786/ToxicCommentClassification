import pandas as pd
import re


df = pd.read_csv("data/raw/train.csv")

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# print(df.columns)

# print(df["comment_text"].head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

df["clean_text"] = df["comment_text"].apply(clean_text)

# print(df["clean_text"].head())

df = df.drop(columns=["comment_text"])

print(df.columns)
df = df.to_csv("data/processed/clean_train.csv", index=False)



