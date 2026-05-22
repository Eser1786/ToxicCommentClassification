from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

    return mlb_model, X_test, y_test_ml, y_pred_ml, y_prob_ml, cols

def train_multi_labels_model(X_train, X_test, y_train, y_test):

    cols = ["toxic", "server_toxic", "obscene", "threat", "insult", "identity_hate"]

    y_train_ml = y_train.values
    y_test_ml = y_test.values

    mlb_model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000)
    )

    mlb_model.fit(X_train, y_train_ml)

    y_pred_ml = mlb_model.predict(X_test)
    y_prob_ml = mlb_model.predict_proba(X_test)[:,1]
    y_prob_2d = mlb_model.predict_proba(X_test)

    return mlb_model, X_test, y_test_ml, y_pred_ml, y_prob_ml, y_prob_2d, cols