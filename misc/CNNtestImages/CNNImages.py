import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Define transforms
trainTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
])

testTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),             
])

#Use the built in ImageFolder to load the data, handles labeling automatically
trainingSet = torchvision.datasets.ImageFolder(root='misc/imageDataset/training', transform=trainTransform)
testingSet = torchvision.datasets.ImageFolder(root='misc/imageDataset/testing', transform=testTransform)

#Data loaders
trainingLoader = DataLoader(trainingSet, batch_size=16, shuffle=True)
testingLoader = DataLoader(testingSet, batch_size=16, shuffle=True)

#CNN as defined in the pytorch tutorial
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(132608, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 10

#Training loop
for epoch in range(epochs):
    for i, data in enumerate(trainingLoader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#torch.save(cnn, 'misc/models/imagecnn.pth') #Save the model

model_scripted = torch.jit.script(cnn) # Export to TorchScript
model_scripted.save('misc/models/imagecnn.pt') # Save

"""
correct = 0
total = 0

#Testing loop
with torch.no_grad():
    for data in testingLoader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct // total} %')
"""

