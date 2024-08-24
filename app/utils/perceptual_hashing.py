import numpy as np
import cv2
import pywt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perceptual_hash(image: np.ndarray, hash_size: int = 16) -> str:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to a fixed size
    resized = cv2.resize(gray, (hash_size, hash_size))
    
    # Compute the DCT transform
    dct = cv2.dct(np.float32(resized))
    
    # Extract the top-left 8x8 DCT coefficients
    dct_low = dct[:8, :8]
    
    # Compute the median value
    med = np.median(dct_low)
    
    # Generate the hash
    hash_value = ''
    for i in range(8):
        for j in range(8):
            hash_value += '1' if dct_low[i, j] > med else '0'
    
    logger.debug(f"Generated hash: {hash_value}")
    
    return hash_value