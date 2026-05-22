from train_test_split import load_data
from tfidf_vectorize import vectorize_data
from train_model import train_model_LogisticRegression, train_model_Linear_SVM
from evaluate_model import evaluate_model

X_train, X_test, y_train, y_test = load_data()

X_train, X_test = vectorize_data(X_train, X_test)

# model, X_test, y_test, y_pred, y_prod = train_model_LogisticRegression(X_train, X_test, y_train, y_test)

model, X_test, y_test, y_pred, y_prod = train_model_Linear_SVM(X_train, X_test, y_train, y_test)

evaluate_model(y_test, y_pred, y_prod)
