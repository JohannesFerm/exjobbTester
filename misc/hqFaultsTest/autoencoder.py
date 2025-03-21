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

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Create a custom dataset
class AudioDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dataDir = dir
        self.audioFiles = [f for f in os.listdir(dir) if f.endswith('.wav')]
        self.transform = transform

    #Next two methods are necessary, inheriting from Dataset

    #Return length of dataset
    def __len__(self):
        return len(self.audioFiles)

    #Returns single item/sample and corresponding label of dataset
    def __getitem__(self, index):
        fileName = self.audioFiles[index]

        data, sr = torchaudio.load(os.path.join(self.dataDir, fileName))

        #Way to handle the last clip from the split being less than 10 seconds
        expectedDur = sr * 5 
        if data.shape[1] < expectedDur:
            return None  

        if self.transform: 
            data = self.transform(data)

        return data

#Class to transform subsets so that two subsets can have different transforms
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = [idx for idx in indices if dataset[idx] is not None]
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        tdata = self.transform(data)
        return tdata

def normalizeSpectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / (std + 1e-6)


#Function to randomly transform the samples, data augmentation
def randomTransform(wave):

    #Applied directly on the audio
    wave = T.Vol(gain=random.uniform(0, 10))(wave) 

    #Convert to spectrogram and apply two more
    spectrogram = MelSpectrogram(n_mels=80)(wave)
    spectrogram = T.TimeMasking(time_mask_param=random.randint(10, 100))(spectrogram) 
    spectrogram = T.FrequencyMasking(freq_mask_param=random.randint(5, 30))(spectrogram) 
    spectrogram = normalizeSpectrogram(spectrogram)

    return spectrogram.squeeze(0) 
        
#Load the full dataset
dataset = AudioDataset("misc/husqvarnaDataSplit/reference")

#Split by index
indices = list(range(len(dataset)))
trainIndices, testIndices = train_test_split(
    indices,
    test_size=0.3, 
    random_state=seed, 
    shuffle=True  
)
#Get the two sets of data, with different transforms
trainDataset = TransformedSubset(dataset, trainIndices, randomTransform)
testDataset = TransformedSubset(dataset, testIndices, lambda data: normalizeSpectrogram(MelSpectrogram(n_mels=80)(data).squeeze(0)))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=0)
testDataloader = DataLoader(testDataset, batch_size=2, shuffle=True, num_workers=0)

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(1, 80, 1201)):
        super(Autoencoder, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        
        dummy_input = torch.zeros(1, *input_shape)
        encoded_size = self.encoder(dummy_input).shape
        self.encoded_channels, self.encoded_height, self.encoded_width = encoded_size[1:]
        bottleneck_dim = self.encoded_channels * self.encoded_height * self.encoded_width

        #Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
            nn.ReLU(),
            nn.Unflatten(1, (self.encoded_channels, self.encoded_height, self.encoded_width))
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

model = Autoencoder()

#Put on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 10

#Train the model
model.train()
for epoch in range(epochs):
    runningLoss = 0.0

    for batch in trainDataloader:
        batch = batch.to(device)
        batch = batch.unsqueeze(1)

        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

    avgLoss = runningLoss / len(trainDataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avgLoss:.4f}")

random_input = torch.randn_like(batch)
random_loss = criterion(model(random_input), random_input)
print(f"Loss on random input: {random_loss.item()}")
