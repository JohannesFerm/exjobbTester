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
import timm 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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
        expectedDur = sr * 5 
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

def normalizeSpectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / (std + 1e-6)


#Function to randomly transform the samples, data augmentation
def randomTransform(wave):
    wave = T.Vol(gain=random.uniform(0, 5))(wave)  

    spectrogram = MelSpectrogram(n_mels=80, hop_length=512)(wave)
    spectrogram = T.TimeMasking(time_mask_param=random.randint(5, 20))(spectrogram)  
    spectrogram = T.FrequencyMasking(freq_mask_param=random.randint(2, 10))(spectrogram)
    spectrogram = normalizeSpectrogram(spectrogram)

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
testDataset = TransformedSubset(dataset, testIndices, lambda data: normalizeSpectrogram(MelSpectrogram(n_mels=80, hop_length=512)(data).squeeze(0)))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

#Simple lstm
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        lstmOut, _ = self.lstm(x)
        lastOut = lstmOut[:, -1, :] 
        out = self.fc(lastOut) 
        return out
    
model = LSTM(80, 256, 3, len(set(dataset.labels)))

#Put on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


epochs = 20 
    
#Train the model
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in trainDataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 2, 1)  #LSTM expects different order than standard

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainDataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

#Test the model
model.eval()
total_loss, correct, total = 0, 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testDataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 2, 1)  #LSTM expects different order than standard        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = total_loss / len(testDataloader)
test_acc = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.labelMap.keys(), yticklabels=dataset.labelMap.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels, all_preds, target_names=dataset.labelMap.keys()))
