from app.utils.image_utils import verify_image_format, process_image, compare_images as compare_images_util
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def verify_image(temp_file_path: str):
    try:
        verify_image_format(temp_file_path)
        image_hash = process_image(temp_file_path)
        return {"image_hash": image_hash}
    except Exception as e:
        logger.error(f"Error verifying image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error verifying image: {str(e)}")

def compare_images(temp_file_path1: str, temp_file_path2: str):
    try:
        verify_image_format(temp_file_path1)
        verify_image_format(temp_file_path2)
        return compare_images_util(temp_file_path1, temp_file_path2)
    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")