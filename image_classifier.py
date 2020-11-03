import cv2
import numpy as np
import os

from utils import *


images = []
games_names = []

#################### LOAD IMAGES

for file in os.listdir(IMAGES_QUERY_PATH):
    current_img = cv2.imread(f"{IMAGES_QUERY_PATH}/{file}", 0)  # Import in gray scale
    images.append(current_img)
    games_names.append(os.path.splitext(file)[0])

print_clr(f"Total classes detected: {len(games_names)}", YELLOW)

#################### STORE IMAGES DESCRIPTORS

descriptors = []
for img in images:
    _, desc = find_descriptors(img)
    descriptors.append(desc)

#################### READ WEB-CAM

cam = cv2.VideoCapture(0)

while True:
    _, imgOriginal = cam.read()
    imgOriginal = cv2.resize(imgOriginal, dsize=(IMG_WIDTH, IMG_HEIGHT))
    imgGray = cv2.cvtColor(imgOriginal.copy(), cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop the video
        break
