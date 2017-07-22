import numpy as np
from sklearn.naive_bayes import GaussianNB

#  X 是特征变量
X = np.array([[-1,-1],[-2,9],[-3,-2],[1,1],[2,1],[-0.8,-1]])
# Y 是标签
Y = np.array([1,1,1,2,2,2])
clf = GaussianNB()
# 训练
clf.fit(X,Y)
GaussianNB()
# 得到预测结果
print(clf.predict([-0.8,-1]))



