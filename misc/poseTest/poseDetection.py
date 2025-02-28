import cv2
import time
import pyrealsense2 as rs #For the intel camera
import numpy as np
import mediapipe as mp
from math import atan2

#https://www.youtube.com/watch?v=06TE_U21FK4 great tutorial

#Define angle limits for what is counted as pointing in a direction
lowerLimitArmpit = 75
upperLimitArmpit = 105
lowerLimitElbow = 160
upperLimitElbow = 180 

#Function to calculate joint angles 
def angle(p1, p2, p3):
    a = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
    a = np.rad2deg(a)
    a = abs(a)
    if a > 180:
        a = 360 - a
    return a

#Function to determine pointing direction from angles, only pointing if one arm is extended (not both)
def pointingDirection(laa, raa, lea, rea):
    if lowerLimitArmpit < laa < upperLimitArmpit and not lowerLimitArmpit < raa < upperLimitArmpit:
        return "left"
    elif lowerLimitArmpit < raa < upperLimitArmpit and not lowerLimitArmpit < laa < upperLimitArmpit:
        return "right"
    else:
        return None

#Basic stuff to set up a pipeline for the intel camera
pipe = rs.pipeline()
cfg = rs.config()
profile = pipe.start()

#Mediapipe setup
mpPose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#Video stream
with mpPose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, static_image_mode=False) as pose:
    while True:
        frames = pipe.wait_for_frames()
        cFrame = frames.get_color_frame()
        cImage = np.asanyarray(cFrame.get_data())
        cImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2BGR) #Convert to rgb
        
        mpRes = pose.process(cImage)

        try:
            landmarks = mpRes.pose_landmarks.landmark
        except:
            continue
        
        mp_drawing.draw_landmarks(cImage, mpRes.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )      

        #Get the mediapipe landmarks
        leftShoulder = np.array([landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y])
        rightShoulder = np.array([landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y])
        leftElbow = np.array([landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y])
        rightElbow = np.array([landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y])
        leftWrist = np.array([landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y])
        rightWrist = np.array([landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y])
        leftHip = np.array([landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y])
        rightHip = np.array([landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y])

        #Calculate four necessary angles
        leftArmpitAngle = angle(leftHip, leftShoulder, leftElbow)
        rightArmpitAngle = angle(rightHip, rightShoulder, rightElbow)
        leftElbowAngle = angle(leftShoulder, leftElbow, leftWrist)
        rightElbowAngle = angle(rightShoulder, rightElbow, rightWrist)

        pointingDir = pointingDirection(leftArmpitAngle, rightArmpitAngle, leftElbowAngle, rightElbowAngle)

        #Run motor depending on the pointing direction
        if pointingDir is not None:
            if(pointingDir == "left"):
                print("Run motor left")
            else:
                print("Run motor right")

        print("Motor not running")
        cv2.imshow('Mediapipe Feed', cImage)

        if cv2.waitKey(1) == ord('q'):
            break

pipe.stop()