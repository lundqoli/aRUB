from torch.utils.data import DataLoader

import torch.nn.functional as F
from aRUB import aRUB
from torchvision import datasets, models, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import time
import torchvision.transforms as transforms

import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x


class DenseNet(nn.Module):

    def __init__(self,n_input):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 200, bias=True),
            nn.ReLU(),
            nn.Linear(200,10,bias=True))

    def forward(self,x):
        z = self.net(torch.flatten(x,1))
        return z

def fgsm_max(model,x,y,epsilon, domain=[-1,1]):

    delta = torch.zeros_like(x, requires_grad=True)

    loss = nn.CrossEntropyLoss()(model(x + delta), y)
    loss.backward()

    pertubation = epsilon * delta.grad.detach().sign()

    xnew = x + pertubation

    ones = torch.ones_like(x)

    #stay between within [domain[0], domain[1]]
    xnew = torch.maximum(domain[0]*ones,xnew)
    xnew = torch.minimum(domain[1]*ones,xnew)

    return xnew.detach()

def compute_accuracy(net, testloader,device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def compute_accuracy_attack(net,testloader,device,epsilon, domain=[-1,1]):
    net.eval()
    correct = 0
    total = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        images = fgsm_max(model=net,x=images,y=labels,epsilon=epsilon,domain=domain)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    return correct / total


# Initial parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 2
lr = 0.001
momentum = 0.9
batch_size = 32

epsilon = 0.0001
norm = "L1"
n_classes = 10


transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR10 dataset for testing
train_data = datasets.CIFAR10(root='data',train=True, transform=transform,download=True)
test_data = datasets.CIFAR10(root='data',train=False,transform=transform)


# Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

net = ConvNet()
net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#net = DenseNet(n_input=3*32*32)



optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

n_train = len(train_dataloader)

# loop over dataset
for epoch in range(epochs):

    running_loss = 0.0

    #criterion = aRUB(epsilon, n_classes, device, norm=norm)
    # uncomment the following line and comment the previous line for normal Cross-entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    # For loop
    for i, data in enumerate(train_dataloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        #loss, net = criterion(labels, inputs, net)
        # uncomment the following line and comment the previous line for normal Cross-entropy loss
        loss = criterion(net(inputs),labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'[{epoch + 1} / {epochs}, {i + 1:5d}/{n_train}] loss: {running_loss / 100:.3f} at {time.ctime()}')
            running_loss = 0.0


print("test accuracy: ",compute_accuracy(net, test_dataloader,device))
print("test adverserial accuracy for epsilon=",epsilon," : ", compute_accuracy_attack(net,test_dataloader,device,epsilon*epochs, domain=[0,1]))

