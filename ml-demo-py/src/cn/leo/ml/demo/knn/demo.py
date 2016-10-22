import numpy
import knn

    
inputX = numpy.array([10.1, 10.2])
dataSet = numpy.array([[1.0, 0.9], [0.8, 1.15], [7.9, 8.0], [8.9, 7.5]])
labels = numpy.array(['A', 'A', 'B', 'B'])
k = 3

print knn.knnAlgo(inputX, dataSet, labels, k)  