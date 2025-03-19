import os
import torch
import torchaudio
import torchaudio.transforms
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import random
from sklearn.model_selection import train_test_split
import torchvision.models as models
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
        self.labelMap = {}
        self.labels = []
        self.audioFiles = []

        #Loop through all folders in the dataset
        for idx, label in enumerate(sorted(os.listdir(dir))):
            labelPath = os.path.join(dir, label)
            if(os.path.isdir(labelPath)):
                self.labelMap[label] = idx
                
                #Loop through all files in subfolder (label) and append
                for file in os.listdir(labelPath):
                    if file.endswith(".wav"): 
                        self.audioFiles.append(os.path.join(labelPath, file))
                        self.labels.append(idx)



        self.transform = transform

    #Next two methods are necessary, inheriting from Dataset

    #Return length of dataset
    def __len__(self):
        return len(self.audioFiles)

    #Returns single item/sample and corresponding label of dataset
    def __getitem__(self, index):
        fileName = self.audioFiles[index]
        label = self.labels[index]

        data, sr = torchaudio.load(fileName)

        #Way to handle the last clip from the split being less than 10 seconds
        expectedDur = sr * 10 
        if data.shape[1] < expectedDur:
            return None  

        if self.transform: 
            data = self.transform(data)

        return data, label

#Class to transform subsets so that two subsets can have different transforms
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = [idx for idx in indices if dataset[idx] is not None]
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

    return spectrogram.squeeze(0) 
        
#Load the full dataset
dataset = AudioDataset("misc/husqvarnaDataSplit/")

#Split dataset by index
labels = torch.tensor(dataset.labels)
trainIndices, testIndices = train_test_split(
    list(range(len(dataset))),
    test_size=0.3,
    stratify=labels, 
    random_state=seed
)

#Get the two sets of data, with different transforms
trainDataset = TransformedSubset(dataset, trainIndices, randomTransform)
testDataset = TransformedSubset(dataset, testIndices, lambda data: MelSpectrogram(n_mels=40)(data).squeeze(0))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=0)
testDataloader = DataLoader(testDataset, batch_size=8, shuffle=True, num_workers=0)

# Load MobileNetV3 (pretrained) and modify it
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
num_classes = len(set(dataset.labels))  
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

#Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

epochs = 20

#Train the model
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in trainDataloader:
        inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step(running_loss / len(trainDataloader))


    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainDataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

model.eval()
total_loss, correct, total = 0, 0, 0

for inputs, labels in testDataloader:
    print(f"Input shape: {inputs.shape}, Labels: {labels}")  # Debug output
    break  # Just check the first batch


#Test the model
with torch.no_grad():
    for inputs, labels in testDataloader:
        inputs = inputs.unsqueeze(1)  
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


print(f"Test Loss: {total_loss/len(testDataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")