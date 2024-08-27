import cv2
import numpy as np
import hashlib
from typing import Union, List

def preprocess_frame(frame: np.ndarray, hash_size: int = 32) -> np.ndarray:
    # Convert to grayscale if it's not already
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame
    frame = cv2.resize(frame, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    
    # Normalize the frame
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    return frame

def perceptual_hash(frame: np.ndarray, hash_size: int = 32) -> str:
    # Preprocess the frame
    processed_frame = preprocess_frame(frame, hash_size)
    
    # Compute the DCT transform
    dct = cv2.dct(np.float32(processed_frame))
    
    # Extract the top-left 8x8 DCT coefficients
    dct_low = dct[:8, :8]
    
    # Compute the median value (excluding the first term)
    median = np.median(dct_low[1:])
    
    # Generate the hash
    hash_value = ''
    for i in range(8):
        for j in range(8):
            hash_value += '1' if dct_low[i, j] > median else '0'
    
    return hash_value

def compute_video_hash(frames: List[np.ndarray]) -> str:
    frame_hashes = [perceptual_hash(frame) for frame in frames]
    combined_hash = ''.join(frame_hashes)
    
    # Use SHA256 to create a fixed-length hash
    return hashlib.sha256(combined_hash.encode()).hexdigest()

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the Hamming distance between two hash strings"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def are_videos_similar(hash1: str, hash2: str, threshold: float = 0.1) -> bool:
    """Determine if two videos are similar based on their hash distance"""
    distance = hamming_distance(hash1, hash2)
    max_distance = len(hash1)
    similarity = 1 - (distance / max_distance)
    return similarity >= (1 - threshold)