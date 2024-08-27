import os
import shutil
from PIL import Image
from fastapi import HTTPException
from app.utils.perceptual_image_hashing import perceptual_image_hash, are_images_similar, hamming_distance
import logging

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

def verify_image_format(filename: str):
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported formats are: {', '.join(SUPPORTED_IMAGE_FORMATS)}")

def process_image(temp_file_path: str):
    try:
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Saved image not found: {temp_file_path}")
        
        # Open and verify the image
        with Image.open(temp_file_path) as img:
            img.verify()  # Verify that it's a valid image file
        
        # Generate perceptual hash for the image
        with Image.open(temp_file_path) as img:
            image_hash = perceptual_image_hash(img)
        
        return image_hash
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def compare_images(temp_file_path1: str, temp_file_path2: str):
    try:
        # Generate perceptual hashes for both images
        with Image.open(temp_file_path1) as img1, Image.open(temp_file_path2) as img2:
            hash1 = perceptual_image_hash(img1)
            hash2 = perceptual_image_hash(img2)
        
        # Compare the hashes
        are_similar = are_images_similar(hash1, hash2)
        distance = hamming_distance(hash1, hash2)
        
        return {
            "image1_hash": hash1,
            "image2_hash": hash2,
            "are_similar": are_similar,
            "hamming_distance": distance
        }
    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")

def cleanup_temp_dirs(temp_dirs):
    for temp_dir in temp_dirs:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)