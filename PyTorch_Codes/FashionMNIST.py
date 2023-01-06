# Multi-Label Image Classification of the Fashion-MNIST dataset

'''
We test the eBrAVO and pBrAVO algorithms for
Multi-Label Image Classification of the Fashion-MNIST dataset

More details can be found in
    "Practical Perspectives on Symplectic Accelerated Optimization"
    Authors: Valentin Duruisseaux and Melvin Leok. 2022.


"Fashion-MNIST is a dataset of Zalando's article images consisting of a
training set of 60,000 examples and a test set of 10,000 examples. Each example
is a 28x28 grayscale image, associated with a label from 10 classes
(t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)."

We learn the 55,050 parameters of a Neural Network classification model.

Usage:

	python ./PyTorch_Codes/FashionMNIST.py

'''


################################################################################

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import BrAVO_torch

from torchsummary import summary


################################################################################

epochs = 20

loss_fn = nn.CrossEntropyLoss()

################################################################################
# Download Training Data
training_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=64)
# Download Testing Data
test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())
test_dataloader = DataLoader(test_data, batch_size=64)


################################################################################
# Construct Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


################################################################################
# Training

def train_loop(dataloader, model, loss_fn, optimizer):

    CollectedLosses = np.array([])

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and Optimization Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect and print the loss every 100 iterations
        if batch % 100 == 0:
            CollectedLosses = np.append(CollectedLosses,loss.item())
            loss = loss.item()
            print(f"loss: {loss:>7f}")

    return CollectedLosses


################################################################################
# Testing

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100*correct




################################################################################
# With ADAM

model = NeuralNetwork()

summary(model, (1, 28, 28))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
ADAM_Losses = np.array([])
ADAM_Accuracies = np.array([])
for t in range(epochs):
    print(f"Adam: Epoch {t+1}\n-------------------------------")
    ADAM_Losses = np.append(ADAM_Losses ,train_loop(train_dataloader, model, loss_fn, optimizer))
    ADAM_Accuracies = np.append(ADAM_Accuracies,test_loop(test_dataloader, model, loss_fn))


################################################################################
# With SGD

model2 = NeuralNetwork()
optimizer2 = torch.optim.SGD(model2.parameters(), lr = 0.1)
SGD_Losses = np.array([])
SGD_Accuracies = np.array([])
for t in range(epochs):
    print(f"SGD: Epoch {t+1}\n-------------------------------")
    SGD_Losses = np.append(SGD_Losses ,train_loop(train_dataloader, model2, loss_fn, optimizer2))
    SGD_Accuracies = np.append(SGD_Accuracies,test_loop(test_dataloader, model2, loss_fn))


################################################################################
# With eBrAVO

model3 = NeuralNetwork()
optimizer3 = BrAVO_torch.eBravo(model3.parameters(), lr = 170, C = 1e-5)
eBrAVO_Losses = np.array([])
eBrAVO_Accuracies = np.array([])
for t in range(epochs):
    print(f"eBrAVO: Epoch {t+1}\n-------------------------------")
    eBrAVO_Losses = np.append(eBrAVO_Losses ,train_loop(train_dataloader, model3, loss_fn, optimizer3))
    eBrAVO_Accuracies = np.append(eBrAVO_Accuracies,test_loop(test_dataloader, model3, loss_fn))


################################################################################
# With pBrAVO

model4 = NeuralNetwork()
optimizer4 = BrAVO_torch.pBravo(model4.parameters(), lr = 0.05, C = 1)
pBrAVO_Losses = np.array([])
pBrAVO_Accuracies = np.array([])
for t in range(epochs):
    print(f"pBrAVO: Epoch {t+1}\n-------------------------------")
    pBrAVO_Losses = np.append(pBrAVO_Losses ,train_loop(train_dataloader, model4, loss_fn, optimizer4))
    pBrAVO_Accuracies = np.append(pBrAVO_Accuracies,test_loop(test_dataloader, model4, loss_fn))


################################################################################
# Plot Loss on Training Set

plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs*np.arange(1,ADAM_Losses.size+1)/ADAM_Losses.size,ADAM_Losses,'black',label="ADAM",linewidth=2)
plt.plot(epochs*np.arange(1,SGD_Losses.size+1)/SGD_Losses.size,SGD_Losses,'green',label="SGD",linewidth=2)
plt.plot(epochs*np.arange(1,eBrAVO_Losses.size+1)/eBrAVO_Losses.size,eBrAVO_Losses,'blue',label="eBrAVO",linewidth=2)
plt.plot(epochs*np.arange(1,pBrAVO_Losses.size+1)/pBrAVO_Losses.size,pBrAVO_Losses,'red',label="pBrAVO",linewidth=2)
plt.xlabel("epochs",fontsize=14)
plt.ylabel("Loss",fontsize=14)
plt.legend(fontsize=14)

# Plot Accuracy on Testing Set

plt.subplot(1, 2, 2)
plt.plot(np.arange(1,epochs+1),ADAM_Accuracies,'black',label="ADAM",linewidth=2)
plt.plot(np.arange(1,epochs+1),SGD_Accuracies,'green',label="SGD",linewidth=2)
plt.plot(np.arange(1,epochs+1),eBrAVO_Accuracies,'blue',label="eBrAVO",linewidth=2)
plt.plot(np.arange(1,epochs+1),pBrAVO_Accuracies,'red',label="pBrAVO",linewidth=2)
plt.xlabel("epochs",fontsize=14)
plt.ylabel("Accuracy",fontsize=14)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()