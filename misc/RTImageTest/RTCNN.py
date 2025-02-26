import numpy as np
import cv2
import pyrealsense2 as rs #For the intel camera
import torch
from torchvision import transforms
import time

#https://www.youtube.com/watch?v=CmDO-w56qso, good tutorial for working with an intel realsense camera in python

#Basic stuff to set up a pipeline for the intel camera
pipe = rs.pipeline()
cfg = rs.config()
profile = pipe.start()

#Load the model
cnn = torch.jit.load("misc/models/imagecnn.pt")
cnn.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224)             
])

while True:

    #Collect frames
    frames = pipe.wait_for_frames()
    cFrame = frames.get_color_frame()
    cImage = np.asanyarray(cFrame.get_data())
    cImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2BGR) #Convert to rgb

    cv2.imshow('camera', cImage)

    #Predict using the CNN
    imageTensor = transform(cImage)
    imageTensor = imageTensor.unsqueeze(0)
    output = cnn(imageTensor)
    _, predicted = torch.max(output, 1)
    print(predicted.item())
    
    if cv2.waitKey(1) == ord('q'):
        break


pipe.stop()