import librosa
import numpy as np
import os
import pandas as pd
import pickle

dir = "misc/mowerDataset/"
clipDuration = 2

labels = os.listdir(dir)

datasetArray = []

for label in labels:
    audioClips = []
    imuChunks = []
    
    #Loop through files, make sure audio file is processed first
    for file in sorted(os.listdir(dir + label), key=lambda x: 0 if ".wav" in x else 1):
        
        #Process audio
        if file.endswith(".wav"):
            audio, sr = librosa.load(dir + label + "/" + file, sr=None)
            numClips = len(audio) // (clipDuration * sr)
            audioClips = [audio[i*clipDuration*sr : (i+1)*clipDuration*sr] for i in range(numClips)]
        
        #Process IMU 
        elif file.endswith(".csv"):
            df = pd.read_csv(dir + label + "/" + file)
            t0 = df["timestamp"][0]
            df["timestamp"] = df["timestamp"].apply(lambda x: (x - t0)) #Offset the time so it begins at 0

            #Chunk IMU data to match the audio length
            imuChunks = [df.loc[(df["timestamp"] >= i * clipDuration) & (df["timestamp"] < (i + 1) * clipDuration), ["roll", "pitch", "yaw"]].values for i in range(len(audioClips))]

    #Create dataset
    seqLength = 5
    for i in range(len(audioClips)):
        audio = librosa.feature.melspectrogram(y=audioClips[i], sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        audio = librosa.power_to_db(audio, ref=np.max)
        imu = imuChunks[i]
        
        if len(imu) < 2:
            #Handle missing IMU data, assume linearity and simply get a new value by averaging
            if len(imu) == 1:
                if i < len(audioClips) - 1:                
                    newData = np.array([np.mean(j) for j in zip(imu[0], imuChunks[i + 1][0])])
                    imu = np.append(imu, [newData], axis=0)
                else:
                    newData = np.array([np.mean(j) for j in zip(imu[0], imuChunks[i - 1][0])])
                    imu = np.append(imu, [newData], axis=0)
            elif len(imu) == 0: #Skip samples where IMU data is empty
                continue
            
        datasetArray.append([audio, imu, label])
    
#Write dataset to file
dataFrame = pd.DataFrame(datasetArray, columns=["audio", "imu", "label"])
dataFrame.to_pickle('misc/mowerModel/mowerData.pkl')

"""
TODO:
Lägg till i slutet att göra IMU-sekvenser på 5 (10 s)
Kolla om det går att köra denna + modell i ett gemensamt skript som också laddar upp modellen på rpin
Ändra paths, kolla i data_collection skriptet
"""