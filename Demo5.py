import random


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]


def euc(a, b):
    return distance.euclidean(a, b)



from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import tree
from scipy.spatial import distance

# from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5)

my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

print(accuracy_score(y_test, predictions))
