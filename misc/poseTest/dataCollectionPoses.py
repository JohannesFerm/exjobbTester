#Basic data collection script for images
#Similar to the other script for image collection in this repo, but with different camera
import cv2
import time
import pyrealsense2 as rs #For the intel camera
import numpy as np
cam = cv2.VideoCapture(0)

#Basic stuff to set up a pipeline for the intel camera
pipe = rs.pipeline()
cfg = rs.config()
profile = pipe.start()

training = input("Training? (0 or 1): ") == '1'
if training:
    filePath = "training"
else:
    filePath = 'testing'

label = input("Image label: ")

i = 0

#Capture frames from the camera and save in correct folder, do 100 at a time
while i < 500:
    frames = pipe.wait_for_frames()
    cFrame = frames.get_color_frame()
    cImage = np.asanyarray(cFrame.get_data())
    cImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2BGR) #Convert to rgb

    fileName = f"image_{int(time.time() * 1000)}_{i+1:03d}.png"

    cv2.imwrite('misc/poseDataset/{0}/{1}/'.format(filePath, label) + fileName, cImage)
    i += 1

cam.release()
