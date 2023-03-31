import torch.utils.data
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class DogCatDataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform

        dog_dir = os.path.join(img_dir, "dog")
        cat_dir = os.path.join(img_dir, "cat")
        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(dog_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))
        random.shuffle(imgsLib)
        self.imgsLib = imgsLib

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.imgsLib[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0  #狗的label设为1，猫的设为0

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgsLib)


if __name__ == "__main__":

    CLASSES = {0: "cat", 1: "dog"}
    img_dir_train = "./data/train"

    data_transform = transforms.Compose([
        transforms.Resize(256),  # resize到256
        transforms.CenterCrop(224),  # crop到224
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor /255.操作

    ])

    train_dataSet = DogCatDataSet(img_dir=img_dir_train, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=8, shuffle=True, num_workers=4)

    img_dir_val = "./data/validation"

    val_dataSet = DogCatDataSet(img_dir=img_dir_val, transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataSet, batch_size=8, shuffle=True, num_workers=4)

    # image_batch, label_batch = iter(dataLoader)

    # for i in range(image_batch.data.shape[0]):
    #     label = np.array(label_batch.data[i])  ## tensor ==> numpy
    #     # print(label)
    #     img = np.array(image_batch.data[i] * 255, np.int32)
    #     print(CLASSES[int(label)])
    #     plt.imshow(np.transpose(img, [1, 2, 0]))
    #     plt.show()
