from torch.utils.data import DataLoader
from aRUB import aRUB
from torchvision import datasets
import torch
import time
import torchvision.transforms as transforms
import torch.nn as nn

def fgsm_max(model,x,y,epsilon, domain):

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

def compute_accuracy_attack(net,testloader,device,epsilon, domain):
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


# Learning parameters
epochs = 10
lr = 0.001
momentum = 0.9
batch_size = 32

# aRUB parameters
epsilon = 0.002
norm = "L1"
n_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR10 dataset for testing
train_data = datasets.CIFAR10(root='data',train=True, transform=transform,download=True)
test_data = datasets.CIFAR10(root='data',train=False,transform=transform)

# Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Load resnet
net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
net.fc = nn.Linear(in_features=512,out_features=10,bias=True)

# Set model to cuda if possible
if torch.cuda.is_available():
    net.cuda()

# Specify optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# get the size of the training set for printouts
n_train = len(train_dataloader)

# Train neural networks
for epoch in range(epochs):

    running_loss = 0.0

    criterion = aRUB(epsilon, n_classes, device, norm=norm)
    # uncomment the following line and comment the previous line for normal Cross-entropy loss
    #criterion = torch.nn.CrossEntropyLoss()

    # For loop
    for i, data in enumerate(train_dataloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        loss, net = criterion(labels, inputs, net)
        # uncomment the following line and comment the previous line for normal Cross-entropy loss
        #loss = criterion(net(inputs),labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'[{epoch + 1} / {epochs}, {i + 1:5d}/{n_train}] loss: {running_loss / 100:.3f} at {time.ctime()}')
            running_loss = 0.0


print("test accuracy: ",compute_accuracy(net, test_dataloader,device))
print("test adverserial accuracy for epsilon=",epsilon," : ", compute_accuracy_attack(net,test_dataloader,device,epsilon*epochs, domain=[-1,1]))
