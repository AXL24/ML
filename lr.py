from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Example:
def train_logistic_regression(filtered_X, filtered_y):
    
    filtered_y = (filtered_y >= 7).astype(int)  # Binary classification: 1 = high quality
    X_train, X_test, y_train, y_test = train_test_split(filtered_X, filtered_y, test_size=0.2, random_state=42, shuffle=True)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return clf, report
