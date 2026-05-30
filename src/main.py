from train_test_split import load_data, load_data_multi_label
from tfidf_vectorize import vectorize_data
from train_model import train_model_LogisticRegression, train_model_Linear_SVM
from evaluate_model import evaluate_model, evaluate_multi_labels_model
from multi_labels import train_multi_labels_model_balanced, train_multi_labels_model
from false_negative_comments import get_false_negative_comment
from save_and_load import load_model, load_tfidf
import pandas as pd


# === sử dụng các model cơ bản ===

# X_train, X_test, y_train, y_test = load_data()

# X_train, X_test = vectorize_data(X_train, X_test)

# # model, X_test, y_test, y_pred, y_prob = train_model_LogisticRegression(X_train, X_test, y_train, y_test)

# model, X_test, y_test, y_pred, y_prob = train_model_Linear_SVM(X_train, X_test, y_train, y_test)

# evaluate_model(y_test, y_pred, y_prob)



# === sử dụng model multi label ===



# có sử dụng balanced
# model, X_test, y_test, y_pred, y_prob, cols = train_multi_labels_model_balanced(X_train, X_test, y_train, y_test) 

# không sử dụng balanced


# print("=== TRƯỚC THRESHOLD TUNING ===")
# evaluate_multi_labels_model(y_test, y_pred_ml, y_prob_2d, cols)
# evaluate_multi_labels_model(y_test, y_pred, y_prob, y_prob_2d, cols)

# for label in cols:
#     get_false_negative_comment(X_test_raw, y_test, y_pred, label, cols)

# sử dụng threshold tuning

# X_train, X_test, y_train, y_test = load_data_multi_label() #data\processed\clean_train.csv

# X_test_raw = X_test

# X_train, X_test = vectorize_data(X_train, X_test)

# model, X_test, y_test_ml, y_pred_ml, y_prob_2d, y_pred_tuned, best_thresholds, cols = train_multi_labels_model(X_train, X_test, y_train, y_test) 

# print ()

# print("=== SAU THRESHOLD TUNING ===")
# evaluate_multi_labels_model(y_test, y_pred_tuned, y_prob_2d, cols)




