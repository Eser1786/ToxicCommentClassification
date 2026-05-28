from train_test_split import train_test_split, load_data, load_data_multi_label
from tfidf_vectorize import TfidfVectorizer,vectorize_data
from train_model import train_model_Linear_SVM, train_model_LogisticRegression
from multi_labels import train_multi_labels_model, train_multi_labels_model_balanced

# X_train, X_test, y_train, y_test = load_data()

# X_train, X_test = vectorize_data(X_train, X_test)

# model, X_test, y_test, y_pred, y_prob = train_model_LogisticRegression(X_train, X_test, y_train, y_test)

# model, X_test, y_test, y_pred, y_prob = train_model_Linear_SVM(X_train, X_test, y_train, y_test)



X_train, X_test, y_train, y_test = load_data_multi_label()

X_test_raw = X_test

X_train, X_test = vectorize_data(X_train, X_test)

# model, X_test, y_test, y_pred, y_prob, cols = train_multi_labels_model_balanced(X_train, X_test, y_train, y_test) 

model, X_test, y_test, y_pred, y_prob, y_prob_2d, cols = train_multi_labels_model(X_train, X_test, y_train, y_test) 