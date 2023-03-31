import gzip
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
MNIST_TRAIN_IMS_GZ = os.path.join(DATASET_DIR, "train-images-idx3-ubyte.gz")
MNIST_TRAIN_LBS_GZ = os.path.join(DATASET_DIR, "train-labels-idx1-ubyte.gz")
MNIST_TEST_IMS_GZ = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
MNIST_TEST_LBS_GZ = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")

NROWS = 28
NCOLS = 28


def load_data():
    print("Unpacking training images ...")
    with gzip.open(MNIST_TRAIN_IMS_GZ, mode='rb') as f:
        magic_num, train_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, train_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz * nrows * ncols, data_bn)
        train_ims = np.asarray(data)
        train_ims = train_ims.reshape(train_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking training labels ...")
    with gzip.open(MNIST_TRAIN_LBS_GZ, mode='rb') as f:
        magic_num, train_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, train_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz, data_bn)
        train_lbs = np.asarray(data)
    print("~" * 5)

    print("Unpacking test images ...")
    with gzip.open(MNIST_TEST_IMS_GZ, mode='rb') as f:
        magic_num, test_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, train_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz * nrows * ncols, data_bn)
        test_ims = np.asarray(data)
        test_ims = test_ims.reshape(test_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking test labels ...")
    with gzip.open(MNIST_TEST_LBS_GZ, mode='rb') as f:
        magic_num, test_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, train_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz, data_bn)
        test_lbs = np.asarray(data)
    print("~" * 5)
    return train_ims, train_lbs, test_ims, test_lbs


train_ims, train_lbs, test_ims, test_lbs = load_data()

mask = np.logical_or(train_lbs == 0, train_lbs == 1)
train_ims = train_ims[mask, :]
train_lbs = train_lbs[mask]

mask = np.logical_or(test_lbs == 0, test_lbs == 1)
test_ims = test_ims[mask, :]
test_lbs = test_lbs[mask]

train_ims = train_ims[:int(0.8 * train_ims.shape[0]), :]
val_ims = train_ims[int(0.8 * train_ims.shape[0]):, :]
train_lbs = train_lbs[:int(0.8 * train_lbs.shape[0])]
val_lbs = train_lbs[int(0.8 * train_lbs.shape[0]):]

test_ims = np.array(test_ims, dtype='float32')
test_lbs = np.array(test_lbs, dtype='float32')
train_ims = np.array(train_ims, dtype='float32')
train_lbs = np.array(train_lbs, dtype='float32')
val_ims = np.array(val_ims, dtype='float32')
val_lbs = np.array(val_lbs, dtype='float32')

test_lbs[test_lbs == 0] = -1
train_lbs[train_lbs == 0] = -1
val_lbs[val_lbs == 0] = -1

weights = np.random.normal(0.0, 1.0, size=(NROWS * NCOLS))

eta = 0.5
for idx in range(train_ims.shape[0]):
    # read the i-th image
    x = train_ims[idx, :]
    # read the i-th label
    y_true = train_lbs[idx]
    y_pred = np.sign(np.dot(x, weights))
    error = y_true - y_pred
    update = eta * error * x
    weights += update
    # every 100 step we want to check the accuracy over the validation data
    acc_count = 0  # we will store the number of correct predictions
    if idx % 100 == 0:
        for val_idx in range(val_ims.shape[0]):
            x = val_ims[val_idx, :]
            y_true = val_lbs[val_idx]
            y_pred = np.sign(np.dot(x, weights))
            if y_true == y_pred:
                acc_count = acc_count + 1
            # predict the label of the sample
            # if prediction is correct, increase the counter
        accuracy = acc_count * 100. / val_ims.shape[0]
        print("step:%d, acc:%.2f" % (idx, accuracy))
        if accuracy > 0.90:
            break
    # if accuracy is above 0.90, terminate by using “break”

# acc_count = 0
# for idx in range(test_ims.shape[0]):
#     # read the i-th image
#     x = test_ims[idx, :]
#     # read the i-th label
#     y_true = test_lbs[idx]
#     y_pred = np.sign(np.dot(x, weights))
#     if y_true == y_pred:
#         acc_count = acc_count + 1
#         # predict the label of the sample
#         # if prediction is correct, increase the counter
# accuracy = acc_count * 100. / test_ims.shape[0]
# print("acc:%.2f" % accuracy)
