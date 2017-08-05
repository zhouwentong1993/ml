from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5)

my_classifier = KNeighborsClassifier()


my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

print(accuracy_score(y_test,predictions))


