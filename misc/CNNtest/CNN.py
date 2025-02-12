import os
import torch
import torchaudio
from torchaudio.transforms import Spectrogram
from torch.utils.data import Dataset, DataLoader, random_split

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
        self.audioFiles = [f for f in os.listdir(dir) if not f.endswith('.txt')] #MER ELEGANT LÖSNING MÖJLIGT?

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
            data = self.transform(data)

        return data, label


#Load the actual dataset, set transform to create spectrograms
transform = Spectrogram()
dataset = AudioDataset("misc/audioDataset/", transform)
print(len(dataset))
print(dataset.audioFiles)

#Split into training and testing, 80/20
trainDataset, testDataset = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=True)

