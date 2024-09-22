import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

if __name__ == '__main__':
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)
    y_pred = logistic_model.predict(X)
    print("Prediction on training set:", y_pred)
    print("Accuracy on training set:", logistic_model.score(X, y))