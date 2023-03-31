import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

file_path = './Q1-Data(1).xlsx'
data = pd.read_excel(file_path, header=0)
data = data.values
num_row = data.shape[0]
dataset = data[0:num_row - 1]


# caculate the euclidean distance
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids
        squaredDiff = diff ** 2  #
        squaredDist = np.sum(squaredDiff, axis=1)  # axis=1
        distance = squaredDist ** 0.5  # sqrt
        clalist.append(distance)
    clalist = np.array(clalist)  # Size: len(dateSet)*k  / Contain the distance
    return clalist


def classify(dataSet, centroids, k):
    # the distance to the sample and the centroids
    clalist = calcDis(dataSet, centroids, k)
    # classify and calculate the new centroids
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet) groupby(min) mean()
    newCentroids = newCentroids.values

    # the change
    changed = newCentroids - centroids

    return changed, newCentroids


def kmeans(dataSet, k):
    # randomly pick the centroids
    index = random.sample(range(0, 200), k)
    centroids = []
    for i in range(0, k):
        centroids.append(dataSet[index[i]])

    # update and the change is zero
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
    centroids = sorted(newCentroids.tolist())  # tolist()

    # calculate the cluster
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # caculate the euclidean distance
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()
        cluster[j].append(dataSet[i])

    return centroids, cluster


if __name__ == '__main__':

    centroids, cluster = kmeans(dataset, 5)
    print('Centroids：%s' % centroids)
    print('Clusters：%s' % cluster)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='Samples')
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='Centroids')
    plt.show()
