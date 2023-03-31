import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def computeCost(X, y):
    inner = np.power((X - y), 2)
    return np.sum(inner) / (2 * len(X))


lr = 0.5
x = [[1, -1], [0, 0], [-1, 1], [3, 2], [2, 2], [2, 3]]
y = [0, 0, 0, 1, 1, 1]
w = np.random.normal([1, 2])  # intitial value
w0 = w
bias = np.random.normal([1, ])
bias0 = bias
x = np.array(x)
y = np.array(y)
w = np.array(w)
epoches = 50

for epoch in range(epoches):
    X = np.matmul(x, w.T) + bias
    predict = step_function(X)
    cost = computeCost(predict, y)
    w = w - lr * (np.sum(predict - y.T)) / (2 * len(predict))
    bias = bias - lr * (np.sum(predict - y.T)) / (2 * len(predict))
    print(f'epoch:{epoch + 1}, cost={cost:.2f}')

plt.plot([4, (-bias - 4 * w[0]) / w[1]], [(-bias - 4 * w[1]) / w[0], 4])
plt.scatter(1, -1, color='b')
plt.scatter(0, 0, color='b')
plt.scatter(-1, 1, color='b')
plt.scatter(3, 2, color='r')
plt.scatter(2, 2, color='r')
plt.scatter(2, 3, color='r')
plt.show()
