import cv2
import numpy as np
import imagehash
from PIL import Image
import logging

def compute_video_hash(features):
    logging.info("Computing video hash.")
    return imagehash.phash(Image.fromarray(np.mean(features, axis=0).reshape(8, 8)))

def compute_frame_hashes(cap):
    logging.info("Computing frame hashes")
    frame_hashes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_hash = imagehash.average_hash(Image.fromarray(gray))
        frame_hashes.append(str(img_hash))
    logging.info("Finished computing frame hashes.")
    return frame_hashes