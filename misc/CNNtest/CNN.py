import os
import torch
import torchaudio
import torchaudio.transforms
from torchaudio.transforms import Spectrogram
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio.transforms as T
import random
import torch.nn as nn

#Create a custom dataset
class AudioDataset(Dataset):
    def __init__(self, dir, transform=None):

        self.dataDir = dir

        #Read the labels and store as ints
        with open("misc/audioDataset/labels.txt", "r") as f: #FIXA, SER SLARVIGT UT
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

        data, sRate = torchaudio.load(os.path.join(self.dataDir, fileName))

        if self.transform: 
            data = self.transform(data, sRate)

        return data, sRate, label

#Class to transform subsets so that two subsets can have different transforms
class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        data, sRate, label = self.dataset[self.indices[idx]]
        data = self.transform(data, sRate)
        return data, label

#Function to randomly transform the samples, data augmentation
def randomTransform(wave, sRate):

    #Applied directly on the audio
    wave = T.Vol(gain=random.uniform(-5, 5))(wave) 
    wave = T.PitchShift(sRate, n_steps=random.uniform(-2, 2))(wave)  

    #Convert to spectrogram and apply two more
    spectrogram = Spectrogram()(wave)
    spectrogram = T.TimeMasking(time_mask_param=random.randint(10, 100))(spectrogram) 
    spectrogram = T.FrequencyMasking(freq_mask_param=random.randint(5, 30))(spectrogram) 

    return spectrogram

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        

#Load the full dataset
dataset = AudioDataset("misc/audioDataset/")

#split by index, to workaround that random_split subsets point to the same parent, 70/30
n = len(dataset)
indices = list(range(n))
random.shuffle(indices)
split = int(0.7 * n)
trainIndices = indices[:split]
testIndices = indices[split:]

#Get the two sets of data, with different transforms
trainDataset = TransformedSubset(dataset, trainIndices, randomTransform)
testDataset = TransformedSubset(dataset, testIndices, lambda data, sRate: Spectrogram()(data))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=True)



