from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = './Q1-Data(1).xlsx'
data = pd.read_excel(file_path, header=0)
data = data.values
num_row = data.shape[0]
dataset = data[0:num_row - 1]

gm = GaussianMixture(n_components=5, random_state=0).fit(dataset)
labels = gm.predict(dataset)

for i in range(len(dataset)):
    if labels[i] == 0:
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green')
    if labels[i] == 1:
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='red')
    if labels[i] == 2:
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='black')
    if labels[i] == 3:
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='blue')
    if labels[i] == 4:
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='yellow')
