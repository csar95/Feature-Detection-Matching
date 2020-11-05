import cv2
import numpy as np
import os

from utils import *


game_images = []
games_names = []

#################### LOAD IMAGES

for file in os.listdir(IMAGES_QUERY_PATH):
    current_img = cv2.imread(f"{IMAGES_QUERY_PATH}/{file}", 0)  # Import in gray scale
    game_images.append(current_img)
    games_names.append(os.path.splitext(file)[0])

print_clr(f"Total classes detected: {len(games_names)}", YELLOW)

#################### STORE GAMES DESCRIPTORS

games_descriptors = []
for game_img in game_images:
    _, desc = find_descriptors(game_img)
    games_descriptors.append(desc)

#################### MATCH DESCRIPTORS IN BOTH IMAGES

def find_game(input_img):
    _, input_desc = find_descriptors(input_img)
    bf = cv2.BFMatcher()  # Brute force matcher

    max_matches = 0
    game_id = 0

    try:
        for desc_id, game_desc in enumerate(games_descriptors):
            matches = bf.knnMatch(game_desc, input_desc, k=2)

            num_good_matches, _ = count_good_matches(matches)

            if num_good_matches > max_matches:
                max_matches = num_good_matches
                game_id = desc_id
    except:
        pass

    return game_id if max_matches > NUM_MATCHES_THRESHOLD else None

#################### READ WEB-CAM

cam = cv2.VideoCapture(0)

while True:
    _, imgOriginal = cam.read()
    imgOriginal = cv2.resize(imgOriginal, dsize=(IMG_WIDTH, IMG_HEIGHT))
    imgGray = cv2.cvtColor(imgOriginal.copy(), cv2.COLOR_BGR2GRAY)

    # Based on the degree of similarity detected between matches, declare the game ID
    game_detected = find_game(imgGray)

    if game_detected is not None:
        cv2.putText(imgOriginal, games_names[game_detected], org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    cv2.imshow("Input", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop the video
        break
