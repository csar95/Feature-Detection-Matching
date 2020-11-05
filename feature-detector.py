import cv2
import numpy as np

from utils import *


img_query = cv2.imread(f"{IMAGES_QUERY_PATH}/XBOX Kinect Sports.jpg", 0)  # Read in gray scale
img_train = cv2.imread(f"{IMAGES_TRAIN_PATH}/Kinect.jpg", 0)

# Find keypoints and descriptors for both game_images
kp_query, des_query = find_descriptors(img_query)
kp_train, des_train = find_descriptors(img_train)

#################### VISUALIZE KEYPOINTS DETECTED

img_kp_query = cv2.drawKeypoints(img_query, kp_query, None)
img_kp_train = cv2.drawKeypoints(img_train, kp_train, None)

cv2.imshow("Key-points query image", img_kp_query)
cv2.imshow("Key-points train image", img_kp_train)

#################### MATCH DESCRIPTORS IN BOTH IMAGES (Goal: See how much similary is detected)

# Initialize the BF (Brute Force) matcher
bf = cv2.BFMatcher()

# Find matches between the two game_images based on the descriptors previously detected
matches = bf.knnMatch(des_query, des_train, k=2)

# Get the number of good matches between the two game_images
num_good_matches, good_matches = count_good_matches(matches)

# Based on the number of good matches we can decide whether an image contains the object we are aiming to detect
print_clr(f"Number of good matches: {num_good_matches}", YELLOW)

# Visualize the good matches detected
img_matches = cv2.drawMatchesKnn(img_query, kp_query, img_train, kp_train, good_matches, None, flags=2)  # flags: How do we want to show the matches
cv2.imshow("Image w/ matches", img_matches)

# cv2.imshow("Query image", img_query)
# cv2.imshow("Train image", img_train)
cv2.waitKey(0)
