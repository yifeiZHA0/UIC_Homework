import pandas as pd
import numpy as np


def MSE(predict: np.ndarray, true: np.ndarray):
    result = np.power(predict - true, 2).mean()
    return result


if __name__ == '__main__':
    data = pd.read_csv('./HistoricalQuotes.csv')
    data_list = []
    for item in data['Close/Last']:
        data_list.append(float(item.split('$')[-1].split(' ')[0]))
    data_list.reverse()
    observe = np.array(data_list)
    prediction = np.zeros(observe.shape)
    order = 3
    bias = 0
    w = np.ones(order) / order
    for i in range(observe.size - order):
        data_in = observe[i:i + order]
        prediction[i + order] = np.matmul(data_in, w)+bias
        error = observe[i + order] - prediction[i + order]
        lr = 1 / np.power(data_in, 2).sum()
        w = w + lr * error * data_in
        bias = bias + lr * error

    loss = MSE(prediction[order:], observe[order:])
    print(prediction)
    print(loss)
# print(data_numpy)
