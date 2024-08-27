import cv2
import numpy as np
from typing import Union
import piexif
from PIL import Image
import io

def preprocess_image(image: Union[str, np.ndarray, Image.Image], hash_size: int = 32) -> np.ndarray:
    if isinstance(image, str):
        # If image is a file path, read it and remove metadata
        with open(image, 'rb') as f:
            img = Image.open(f)
            img = strip_metadata(img)
            image = np.array(img)
    elif isinstance(image, Image.Image):
        image = strip_metadata(image)
        image = np.array(image)
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize the image
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    
    # Normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image

def strip_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    img_without_exif = Image.new(img.mode, img.size)
    img_without_exif.putdata(data)
    return img_without_exif

def perceptual_image_hash(image: Union[str, np.ndarray, Image.Image], hash_size: int = 32) -> str:
    # Preprocess the image
    processed_image = preprocess_image(image, hash_size)
    
    # Compute the DCT transform
    dct = cv2.dct(np.float32(processed_image))
    
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

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the Hamming distance between two hash strings"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def are_images_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """Determine if two images are similar based on their hash distance"""
    distance = hamming_distance(hash1, hash2)
    return distance <= threshold