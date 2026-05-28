from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from save_and_load import save_model


def train_multi_labels_model_balanced(X_train, X_test, y_train, y_test):

    cols = ["toxic", "server_toxic", "obscene", "threat", "insult", "identity_hate"]

    y_train_ml = y_train.values
    y_test_ml = y_test.values

    mlb_model = OneVsRestClassifier(
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )

    mlb_model.fit(X_train, y_train_ml)

    y_pred_ml = mlb_model.predict(X_test)
    y_prob_ml = mlb_model.predict_proba(X_test)[:,1]

    save_model(mlb_model, "multi_labels_balanced")

    return mlb_model, X_test, y_test_ml, y_pred_ml, y_prob_ml, cols

def train_multi_labels_model(X_train, X_test, y_train, y_test):

    cols = ["toxic", "server_toxic", "obscene", "threat", "insult", "identity_hate"]

    y_train_ml = y_train.values
    y_test_ml = y_test.values

    # counts = y_test_ml.sum(axis=0)
    # df_check= pd.DataFrame({
    #     "label": cols,
    #     "count": counts
    # })

    # print(df_check.to_string(index=False))



    mlb_model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000)
    )

    mlb_model.fit(X_train, y_train_ml)

    y_pred_ml = mlb_model.predict(X_test)
    y_prob_ml = mlb_model.predict_proba(X_test)[:,1]
    y_prob_2d = mlb_model.predict_proba(X_test)

    save_model(mlb_model, "multi_labels")

    return mlb_model, X_test, y_test_ml, y_pred_ml, y_prob_ml, y_prob_2d, cols