#Basic data collection script for images

import cv2
import time

cam = cv2.VideoCapture(0)

training = input("Training? (0 or 1): ") == '1'
if training:
    fpath = "training"
else:
    fpath = 'testing'

label = input("Image label: ")

i = 0

#Capture frames from the camera and save in correct folder, do 50 at a time
while i < 50:
    ret, frame = cam.read()
    file_name = f"image_{int(time.time() * 1000)}_{i+1:03d}.png"

    cv2.imwrite('misc/imageDataset/{0}/{1}/'.format(fpath, label) + file_name, frame)
    i += 1
    time.sleep(0.1) #To not save images too fast, not sure if necessary atm

cam.release()
