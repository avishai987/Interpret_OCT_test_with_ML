# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:14:18 2021

@author: Lital Barak and Avishai Wizel
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision    
import os
from math import floor, ceil
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
seed = 1
torch.manual_seed(seed)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import os.path


import torch

    
import torch; torch.manual_seed(0)
import torch.utils
import torch.distributions
#import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import datasets, transforms
#%% load data

transform = transforms.Compose([transforms.CenterCrop(64),
                                transforms.ToTensor()])

data_dir  = "C:/Users/avish/CLionProjects/IP data/OCT2017/OCT2017/"

dataset_train = datasets.ImageFolder(root=data_dir+"train", transform=transform)
dataset_test = datasets.ImageFolder(root=data_dir+"test", transform=transform)


train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False)

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1,padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3= nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6, stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout()
        )
        self.layer4= nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x =F.relu(self.fc2(x))
        x =F.relu(self.fc3(x))
        x =F.relu(self.fc4(x))

        return torch.nn.functional.log_softmax(x, -1)

def train (epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate (train_loader):
        data, labels = data, labels
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,labels)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
             epoch, batch_idx * len(data), len(train_loader)*64,
            100. * batch_idx / len(train_loader), loss.item()))


def validation (valid_loader):
        model.eval()
        test_loss = 0
        correct = 0
        predictions_list = []
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data, target
                output = model(data)
                #save predictions
                for i in range (8):
                    prediction = torch.argmax(output, 1)[i].numpy()
                    predictions_list.append(prediction)
                test_loss += F.nll_loss(output,target,reduction= 'sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).cpu().sum()

    
            print(
                f'===========================\nTest validatation set: Average loss: {test_loss/len(dataset_test):.4f}, Accuracy: {correct}/{(len(dataset_test))} '
                f'({100. * correct / len(dataset_test):.0f}%)')
        return predictions_list

#%% main
from sklearn.metrics import f1_score

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range (10):
        train(epoch,model)
        y_pred = validation(test_loader)
f1_score(dataset_test.targets,y_pred,average = 'weighted')
        
#%%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(dataset_test.targets, y_pred)


#%%
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

y_test = label_binarize(dataset_test.targets, classes=[0,1,2,3])
y_score = label_binarize(y_pred, classes=[0,1,2,3])
n_classes = 4
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + dataset_train.classes[i])
    plt.legend(loc="lower right")
    plt.show()


