import os
import torch
import torchaudio
import torchaudio.transforms
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

seed = 42
random.seed(seed)
torch.manual_seed(seed)

#Create a custom dataset
class AudioDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dataDir = dir

        #Read the labels and store as ints
        with open(dir + "/labels.txt", "r") as f:
            labelString = f.read()
        labelStringArr = list(labelString)
        self.labels = [int(s) for s in labelStringArr]

        #Save file names in sorted order so they correspond to the labels
        self.audioFiles = [f for f in os.listdir(dir) if not f.endswith('.txt')]
        self.audioFiles.sort(key = lambda x: int(x[6:]))

        self.transform = transform

    #Next two methods are necessary, inheriting from Dataset

    #Return length of dataset
    def __len__(self):
        return len(self.audioFiles)

    #Returns single item/sample and corresponding label of dataset
    def __getitem__(self, index):
        fileName = self.audioFiles[index]
        label = self.labels[index]

        data, _ = torchaudio.load(os.path.join(self.dataDir, fileName))

        if self.transform: 
            data = self.transform(data)

        return data, label

#Class to transform subsets so that two subsets can have different transforms
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        tdata = self.transform(data)
        return tdata, label

#Function to randomly transform the samples, data augmentation
def randomTransform(wave):

    #Applied directly on the audio
    wave = T.Vol(gain=random.uniform(0, 10))(wave) 

    #Convert to spectrogram and apply two more
    spectrogram = MelSpectrogram(n_mels=40)(wave)
    spectrogram = T.TimeMasking(time_mask_param=random.randint(10, 100))(spectrogram) 
    spectrogram = T.FrequencyMasking(freq_mask_param=random.randint(5, 30))(spectrogram) 

    return spectrogram

#CNN as defined in the pytorch tutorial
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(18144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x).squeeze(1)
        
#Load the full dataset
dataset = AudioDataset("misc/audioDataset/")

#Split by index, to workaround that random_split subsets point to the same parent, 70/30
n = len(dataset)
indices = list(range(n))
random.shuffle(indices)
split = int(0.7 * n)
trainIndices = indices[:split]
testIndices = indices[split:]

#Get the two sets of data, with different transforms
trainDataset = TransformedSubset(dataset, trainIndices, randomTransform)
testDataset = TransformedSubset(dataset, testIndices, lambda data: MelSpectrogram(n_mels=40)(data))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=0)
testDataloader = DataLoader(testDataset, batch_size=8, shuffle=True, num_workers=0)

#Create a CNN
cnn = CNN()

#Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

epochs = 5 

#Training loop
for epoch in range(epochs):
    for i, data in enumerate(trainDataloader, 0):
        inputs, labels = data
        labels = labels.float()

        optimizer.zero_grad()

        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

correct = 0
total = 0

#Testing loop
with torch.no_grad():
    for data in testDataloader:
        images, labels = data
        outputs = cnn(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct // total} %')



