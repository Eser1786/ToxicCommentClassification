from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from save_and_load import save_model
import numpy as np
from sklearn.metrics import f1_score    


def train_multi_labels_model_balanced(X_train, X_test, y_train, y_test):

    cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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

    cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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
    y_prob_2d = np.column_stack([clf.predict_proba(X_test)[:,1] for clf in mlb_model.estimators_])

    # Threshold tuning
    best_thresholds = {}
    thresholds = np.arange(0.1, 0.9, 0.05)

    for i, label in enumerate(cols):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred_thresh = (y_prob_2d[:, i] >= thresh).astype(int)
            f1 = f1_score(y_test_ml[:, i], y_pred_thresh, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds[label] = best_thresh
        print(f"{label}: threshold = {best_thresh:.2f}, F1 = {best_f1:.3f}")

    # predict lại với threshold tối ưu
    y_pred_tuned = np.zeros_like(y_prob_2d, dtype=int)
    for i, label in enumerate(cols):
        y_pred_tuned[:, i] = (y_prob_2d[:, i] >= best_thresholds[label]).astype(int)

    save_model(mlb_model, "multi_labels")

    return mlb_model, X_test, y_test_ml, y_pred_ml, y_prob_2d, y_pred_tuned, best_thresholds, cols