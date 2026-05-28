# logistic regression
from sklearn.linear_model import LogisticRegression


#linear svm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# save model 
from save_and_load import save_model




def train_model_LogisticRegression(X_train, X_test, y_train, y_test):
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prod = model.predict_proba(X_test)[:,1]

    save_model(model, "logistic_regression")

    return model, X_test, y_test, y_pred, y_prod

def train_model_Linear_SVM(X_train, X_test, y_train, y_test):
    svm = LinearSVC(class_weight="balanced",max_iter=2000)
    model = CalibratedClassifierCV(svm)
    model.fit(X_train, y_train)

    y_pred_svm = model.predict(X_test)
    y_prod_svm = model.predict_proba(X_test)[:,1]

    save_model(model, "linear_svm")

    return model, X_test, y_test, y_pred_svm, y_prod_svm