import librosa
import numpy as np
import os
import pandas as pd

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
            df["timestamp"] = df["timestamp"].apply(lambda x: (x - t0))

            #Chunk IMU data to match the audio length
            imuChunks = [df.loc[(df["timestamp"] >= i * clipDuration) & (df["timestamp"] < (i + 1) * clipDuration), ["roll", "pitch", "yaw"]].values for i in range(len(audioClips))]
    
    #Create dataset
    for i in range(len(audioClips)):
        audio = audioClips[i]
        imu = imuChunks[i]
        
        if len(imu) < 2:
            #Handle missing IMU data, assume linearity and simply get a new value by averaging
            if len(imu) == 1:                
                newData = np.array([np.mean(j) for j in zip(imu[0], imuChunks[i + 1][0])])
                imu = np.append(imu, [newData], axis=0)
            elif len(imu) == 0: #Fix
                print("HEJ")
        datasetArray.append([audio, imu, label])
    

#Write dataset to csv file
dataset = pd.DataFrame(data=datasetArray, columns=["audio", "imu", "label"])
dataset.to_csv('out.csv', index=False)

"""
TODO:
Fixa len(imu) == 0
Fixa ändarna, om len(imu) == 1 i slutet går ej i + 1
Kolla efter så att alla imu arrays har dim 2
"""