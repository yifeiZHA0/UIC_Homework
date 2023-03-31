import torch
import torchvision
import torch.nn as nn
from PIL import Image
import os
import DataLoader
import numpy as np
import torch.utils.data
import os, random, glob
from torchvision import transforms
from PIL import Image


class NeuralNetwork(nn.Module):
    def __init__(self, num_class=2):
        super(NeuralNetwork, self).__init__()
        self.num_class = num_class
        self.base = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')

    def forward(self, x):
        h = self.base(x)
        y = self.fc(h)
        return y

if __name__ == "__main__":
    model = NeuralNetwork(2)

    epochs = 50
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(lr=0.01, params=model.parameters())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)
    opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    max_acc = 0
    epoch_acc = []
    epoch_loss = []
    for epoch in range(epochs):
        for type_id, loader in enumerate([DataLoader.train_loader, DataLoader.val_loader]):
            mean_loss = []
            mean_acc = []
            for images, labels in loader:
                if type_id == 0:
                    # opt_step.step()
                    model.train()
                else:
                    model.eval()
                images = images.to(device)
                labels = labels.to(device).long()
                opt.zero_grad()
                with torch.set_grad_enabled(type_id == 0):
                    outputs = model(images)
                    _, pre_labels = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                if type_id == 0:
                    loss.backward()
                    opt.step()
                acc = torch.sum(pre_labels == labels) / torch.tensor(labels.shape[0], dtype=torch.float32)
                mean_loss.append(loss.cpu().detach().numpy())
                mean_acc.append(acc.cpu().detach().numpy())
            if type_id == 1:
                epoch_acc.append(np.mean(mean_acc))
                epoch_loss.append(np.mean(mean_loss))
                if max_acc < np.mean(mean_acc):
                    max_acc = np.mean(mean_acc)
            print(type_id, np.mean(mean_loss), np.mean(mean_acc))
    print(max_acc)


