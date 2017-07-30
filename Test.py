import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
test_idx = [0, 3, 6]
np.delete(arr, test_idx)
print(arr)
