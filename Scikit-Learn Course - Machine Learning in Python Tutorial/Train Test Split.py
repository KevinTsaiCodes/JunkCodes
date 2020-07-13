from sklearn import datasets
import numpy as np

from sklearn.model_selection import train_test_split
# split it in features and labels
# Split arrays or matrices into random train and test subsets
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

# hours of study vs good/bad grades
# 10 different students
# train with 8 students
# predict with the remaining 2
# level of accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Shape of X/y * test_size__rate == X/y_test
# X/Y_test + X/y_train = X/y
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
