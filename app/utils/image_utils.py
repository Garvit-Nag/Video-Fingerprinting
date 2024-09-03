import os
import cv2
import numpy as np
from typing import Union
from PIL import Image
from io import BytesIO

# Creating a BytesIO object
buffer = BytesIO()

# Writing binary data to the buffer
buffer.write(b"Hello, this is some binary data!")

# Moving the cursor to the beginning of the buffer
buffer.seek(0)

# Reading data from the buffer
data = buffer.read()

print(data)  # Output: b'Hello,

import imghdr
from fastapi import HTTPException

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

def verify_image_format(filename: str):
    file_ext = os.path.splitext(filename)[1].lower()
    
    if not file_ext:
        # If there's no extension, try to determine the image type
        with open(filename, 'rb') as f:
            file_ext = '.' + (imghdr.what(f) or '')
    
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported formats are: {', '.join(SUPPORTED_IMAGE_FORMATS)}")

def preprocess_image(image: Union[str, np.ndarray, Image.Image], hash_size: int = 32) -> np.ndarray:
    if isinstance(image, str):
        with open(image, 'rb') as f:
            img = Image.open(f)
            img = strip_metadata(img)
            image = np.array(img)
    elif isinstance(image, Image.Image):
        image = strip_metadata(image)
        image = np.array(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image

def strip_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    img_without_exif = Image.new(img.mode, img.size)
    img_without_exif.putdata(data)
    return img_without_exif

def perceptual_image_hash(image: Union[str, np.ndarray, Image.Image], hash_size: int = 32) -> str:
    processed_image = preprocess_image(image, hash_size)
    dct = cv2.dct(np.float32(processed_image))
    dct_low = dct[:8, :8]
    median = np.median(dct_low[1:])
    
    hash_value = ''
    for i in range(8):
        for j in range(8):
            hash_value += '1' if dct_low[i, j] > median else '0'
    
    return hash_value

def hamming_distance(hash1: str, hash2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def are_images_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    distance = hamming_distance(hash1, hash2)
    return distance <= threshold

def process_image(temp_file_path: str):
    try:
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Saved image not found: {temp_file_path}")
        
        img = Image.open(BytesIO(temp_file_path))
        img.verify()
        image_hash = perceptual_image_hash(img)
        
        return image_hash
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def compare_images(temp_file_path1: str, temp_file_path2: str):
    try:
        img1 = Image.open(BytesIO(temp_file_path1))
        img2 = Image.open(BytesIO(temp_file_path2))
        hash1 = perceptual_image_hash(img1)
        hash2 = perceptual_image_hash(img2)
        
        are_similar = are_images_similar(hash1, hash2)
        distance = hamming_distance(hash1, hash2)
        
        return {
            "image1_hash": hash1,
            "image2_hash": hash2,
            "are_similar": are_similar,
            "hamming_distance": distance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")