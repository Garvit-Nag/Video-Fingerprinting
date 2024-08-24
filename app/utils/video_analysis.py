import numpy as np
import cv2
import pywt
import moviepy.editor as mp
import logging
from typing import List
import hashlib

def compute_video_hash(frames: List[np.ndarray]) -> str:
    # Concatenate the frames into a single bytes object
    video_bytes = b''.join([cv2.imencode('.png', frame)[1].tobytes() for frame in frames])
    
    # Compute the SHA-256 hash of the video bytes
    video_hash = hashlib.sha256(video_bytes).hexdigest()
    
    return video_hash