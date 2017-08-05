import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

from sklearn.externals.six import StringIO
import pydotplus

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("iris.png")


print(test_data[0], test_target[0])
print(iris.feature_names, iris.target_names)


