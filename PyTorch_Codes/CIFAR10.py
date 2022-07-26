# Multi-Label Image Classification of the CIFAR-10 dataset

'''
We test the eBrAVO and pBrAVO algorithms for
Multi-Label Image Classification of the CIFAR-10 dataset

More details can be found in
    "Practical Perspectives on Symplectic Accelerated Optimization"
    Authors: Valentin Duruisseaux and Melvin Leok. 2022.


"The CIFAR-10 dataset consists of 60000 32x32 colour images in
10 mutually exclusive classes (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck), with 6000 images per class."


We learn the 62,006 parameters of a Convolutional Neural Network
classification model which is very similar to the LeNet-5 architecture
presented in
    “Gradient-based learning applied to document recognition.”
    LeCun, Yann et al.,  Proc. IEEE 86 (1998): 2278-2324.

Usage:

	python ./PyTorch_Codes/CIFAR10.py

'''



################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import BrAVO_torch

from torchsummary import summary


################################################################################
# Load and Prepare CIFAR-10 dataset

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',train=True,transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/',train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=False)
total_step = len(train_loader)


################################################################################
# Model Architecture

class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


################################################################################
# Create Models

epochs = 20

criterion = nn.CrossEntropyLoss()

model1 = LeNet5()
model2 = LeNet5()
model3 = LeNet5()
model4 = LeNet5()

summary(model1, (3, 32, 32))

optimizer1 = torch.optim.Adam(model1.parameters(), lr= 0.008)
optimizer2 = torch.optim.SGD(model2.parameters(), lr= 0.04)
optimizer3 = BrAVO_torch.eBravo(model3.parameters(), lr=0.6, C=100)
optimizer4 = BrAVO_torch.pBravo(model4.parameters(), lr=0.15)


ADAM_Losses = np.array([])
ADAM_Accuracies = np.array([])
SGD_Losses = np.array([])
SGD_Accuracies = np.array([])
eBrAVO_Losses = np.array([])
eBrAVO_Accuracies = np.array([])
pBrAVO_Losses = np.array([])
pBrAVO_Accuracies = np.array([])


################################################################################
# Train and Test the models
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs1 = model1(images)
        outputs2 = model2(images)
        outputs3 = model3(images)
        outputs4 = model4(images)

        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)
        loss3 = criterion(outputs3, labels)
        loss4 = criterion(outputs4, labels)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        loss1.backward()
        loss2.backward()
        loss3.backward()
        loss4.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Test models, store training loss and test accuracy, and print progress
        if (i+1) % 100 == 0:

            model1.eval()
            model2.eval()
            model3.eval()
            model4.eval()

            with torch.no_grad():

                correct1 = 0
                correct2 = 0
                correct3 = 0
                correct4 = 0
                total = 0

                for images, labels in test_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs1 = model1(images)
                    outputs2 = model2(images)
                    outputs3 = model3(images)
                    outputs4 = model4(images)

                    _, predicted1 = torch.max(outputs1.data, 1)
                    _, predicted2 = torch.max(outputs2.data, 1)
                    _, predicted3 = torch.max(outputs3.data, 1)
                    _, predicted4 = torch.max(outputs4.data, 1)

                    total += labels.size(0)

                    correct1 += (predicted1 == labels).sum().item()
                    correct2 += (predicted2 == labels).sum().item()
                    correct3 += (predicted3 == labels).sum().item()
                    correct4 += (predicted4 == labels).sum().item()

                Accuracy1 = 100 * correct1 / total
                Accuracy2 = 100 * correct2 / total
                Accuracy3 = 100 * correct3 / total
                Accuracy4 = 100 * correct4 / total

                ADAM_Accuracies = np.append(ADAM_Accuracies , Accuracy1)
                SGD_Accuracies = np.append(SGD_Accuracies , Accuracy2)
                eBrAVO_Accuracies = np.append(eBrAVO_Accuracies , Accuracy3)
                pBrAVO_Accuracies =np.append(pBrAVO_Accuracies , Accuracy4)

                ADAM_Losses = np.append(ADAM_Losses , loss1.item())
                SGD_Losses = np.append(SGD_Losses , loss2.item())
                eBrAVO_Losses = np.append(eBrAVO_Losses , loss3.item())
                pBrAVO_Losses = np.append(pBrAVO_Losses , loss4.item())

            print("Epoch [{}/{}], Step [{}/{}] ADAM Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(epoch+1, epochs, i+1, total_step, loss1.item(), Accuracy1))
            print("Epoch [{}/{}], Step [{}/{}] SGD Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(epoch+1, epochs, i+1, total_step, loss2.item(), Accuracy2))
            print("Epoch [{}/{}], Step [{}/{}] eBrAVO Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(epoch+1, epochs, i+1, total_step, loss3.item(), Accuracy3))
            print("Epoch [{}/{}], Step [{}/{}] pBrAVO Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(epoch+1, epochs, i+1, total_step, loss4.item(), Accuracy4))
            print('')


################################################################################
# Plot Loss on Training Set

plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(ADAM_Losses,'black',label="ADAM",linewidth=2)
plt.plot(SGD_Losses,'green',label="SGD",linewidth=2)
plt.plot(eBrAVO_Losses,'blue',label="eBrAVO",linewidth=2)
plt.plot(pBrAVO_Losses,'red',label="pBrAVO",linewidth=2)
plt.ylabel("Train Loss",fontsize=14)
plt.legend(fontsize=14)

# Plot Accuracy on Testing Set

plt.subplot(1, 2, 2)
plt.plot(ADAM_Accuracies,'black',label="ADAM",linewidth=2)
plt.plot(SGD_Accuracies,'green',label="SGD",linewidth=2)
plt.plot(eBrAVO_Accuracies,'blue',label="eBrAVO",linewidth=2)
plt.plot(pBrAVO_Accuracies,'red',label="pBrAVO",linewidth=2)
plt.ylabel("Test Accuracy",fontsize=14)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()
