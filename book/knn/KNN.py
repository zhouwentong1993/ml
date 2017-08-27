from numpy import *
import operator
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 将文本记录转换成矩阵

def file2Matrix(fileName):
    fr = open(fileName)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOfLines:
        line = line.strip()
        list_from_line = line.split('\t')
        returnMat[index, :] = list_from_line[0:3]
        classLabelVector.append(int(list_from_line[-1]))
        index += 1

    return returnMat, classLabelVector


def auto_normal(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    normal_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normal_data_set = data_set - tile(min_val, (m, 1))
    normal_data_set = normal_data_set / tile(ranges, (m, 1))
    return normal_data_set, ranges, min_val


def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2Matrix('datingTestSet2.txt')
    normal_mat, ranges, min_val = auto_normal(dating_data_mat)
    m = normal_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0
    for i in range(num_test_vecs):
        classifier_result = classify0(normal_mat[i, :], normal_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m],
                                      3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1

    print(" the total error rate is: %f" % (error_count / float(num_test_vecs)))


dating_class_test()

# dating_data_mat, datingLabels = file2Matrix('datingTestSet2.txt')
# normal_data_set, ranges, min_val = auto_normal(dating_data_mat)
# # print(normal_data_set)
# # print(ranges)
# # print(min_val)
# # print(dating_data_mat)
# # print(datingLabels)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()
