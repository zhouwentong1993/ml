from numpy import *
# print(random.rand(4, 4))

# mat 是生成一个矩阵
# .I 是求逆矩阵
# 矩阵相乘直接使用 * 就行
# eye 代表单位矩阵，通过传入的维度判断是 n 维
m = mat(random.rand(4, 4))
print(m.I * m)

print(eye(4))
a = [1, 2, 3]
# print(a.sum(axis=1))
