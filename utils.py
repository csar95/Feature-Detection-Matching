import cv2


#################### PARAMETERS

IMAGES_QUERY_PATH = "./Resources/ImagesQuery"
IMAGES_TRAIN_PATH = "./Resources/ImagesTrain"
IMG_HEIGHT, IMG_WIDTH = 403, 720
NUM_FEATURES_DETECTOR = 1000
NUM_MATCHES_THRESHOLD = 15

#################### PRINT COLORS

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
GREY = "\x1b[90m"
RESET = "\x1b[0m"


def print_clr(msg, color=RESET):
    print(color + msg + RESET)

def find_descriptors(img):
    # Initialize detector
    orb = cv2.ORB_create(nfeatures=NUM_FEATURES_DETECTOR)  # Other detectors: Swift, Surf

    # Return keypoints and descriptors for the input image
    return orb.detectAndCompute(img, None)  # kp, desc

def count_good_matches(matches):
    good_matches = []

    # Decide what are good matches based on the distance between two features
    for m, n in matches:  # Because k=2, there'll be two values to unpack
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return len(good_matches), good_matches
