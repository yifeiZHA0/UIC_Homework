import numpy as np
import matplotlib.pyplot as plt

# x[n]=7cos(0.1n)+cos(0.95n) the definition of the function

# PartA (2)
x = np.hstack([np.zeros(shape=(15)), np.ones(shape=(15))])
plt.stem(x)
plt.pause(0.0001)
k = np.arange(-10., 20, 1.)
fig2 = plt.figure(2)
plt.stem(k, x)
plt.pause(0.0001)

# PartA (3)
for i in range(-40, 80):
    plt.scatter(i, 7 * np.cos(0.1 * i) + np.cos(0.95 * i), color='r')
plt.pause(0.0001)

for i in range(-40, 80):
    plt.scatter(i + 20, 7 * np.cos(0.1 * i) + np.cos(0.95 * i), color='r')
plt.pause(0.0001)

for i in range(-40, 80):
    plt.scatter(i, np.cos(0.1 * i), color='r')
plt.pause(0.0001)

# PartB (2)

# a

h = 1. / 5. * np.array([1., 1., 1., 1., 1.])
x = np.ones([20, ])
y = np.convolve(x, h, 'same')
for i in range(0, 20):
    plt.scatter(i, y[i], color='r')
plt.pause(0.0001)

# b

x = np.ones([120, ])
for i in range(0, 120):
    x[i] = np.cos(0.1 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)

# c

for i in range(0, 120):
    x[i] = np.cos(0.95 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)

# d

for i in range(0, 120):
    x[i] = 7 * np.cos(0.1 * i) + np.cos(0.95 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)

# PartB (3)

h = [1, -1]

# a

x = np.ones([20, ])
y = np.convolve(x, h, 'same')
for i in range(0, 20):
    plt.scatter(i, y[i], color='r')
plt.pause(0.0001)

# b

for i in range(0, 120):
    x[i] = np.cos(0.1 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)

# c

for i in range(0, 120):
    x[i] = np.cos(0.95 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)

# d

for i in range(0, 120):
    x[i] = 7 * np.cos(0.1 * i) + np.cos(0.95 * i)
y = np.convolve(x, h, 'same')
for i in range(0, 120):
    plt.scatter(i - 40, y[i], color='r')
plt.pause(0.0001)
