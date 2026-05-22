from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_data(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)

    X_train = tfidf.fit_transform(X_train)

    X_test = tfidf.transform(X_test)

    return X_train, X_test
