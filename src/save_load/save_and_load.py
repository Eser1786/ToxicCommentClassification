import joblib


def save_model(model, model_name):
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"{model_name} đã được lưu")

def save_vectorizer(tfidf):
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("vectorizer đã được lưu")

def load_model(model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    print(f"model {model_name} load thành công")
    return model

def load_tfidf():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    print("tfidf load thành công")
    return tfidf