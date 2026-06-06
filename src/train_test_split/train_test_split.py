import pandas as pd

from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/processed/clean_train.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    df["clean_text"] = df["clean_text"].astype(str)
    
    X = df["clean_text"]
    y = df["toxic"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def load_data_multi_label():
    df = pd.read_csv("data/processed/clean_train.csv") # data\processed\clean_train.csv
    df["clean_text"] = df["clean_text"].fillna("")
    df["clean_text"] = df["clean_text"].astype(str)

    X = df["clean_text"]
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

