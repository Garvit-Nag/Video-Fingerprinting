from app.utils.image_utils import verify_image_format, process_image, compare_images as compare_images_util
from fastapi import HTTPException
import logging
from app.utils.file_utils import download_file, remove_temp_file
from app.core.firebase_config import firebase_bucket
from io import BytesIO

logger = logging.getLogger(__name__)

async def verify_image(image_url: str):
    firebase_filename = None
    try:
        firebase_filename = await download_file(image_url)
        verify_image_format(firebase_filename)
        
        # Download the file from Firebase
        blob = firebase_bucket.blob(firebase_filename)
        image_bytes = blob.download_as_bytes()
        
        image_hash = process_image(BytesIO(image_bytes))
        return {"image_hash": image_hash}
    except Exception as e:
        logger.error(f"Error verifying image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error verifying image: {str(e)}")
    finally:
        if firebase_filename:
            await remove_temp_file(firebase_filename)

async def compare_images(image_url1: str, image_url2: str):
    firebase_filename1 = None
    firebase_filename2 = None
    try:
        firebase_filename1 = await download_file(image_url1)
        firebase_filename2 = await download_file(image_url2)
        
        verify_image_format(firebase_filename1)
        verify_image_format(firebase_filename2)
        
        # Download the files from Firebase
        blob1 = firebase_bucket.blob(firebase_filename1)
        blob2 = firebase_bucket.blob(firebase_filename2)
        image_bytes1 = blob1.download_as_bytes()
        image_bytes2 = blob2.download_as_bytes()
        
        return compare_images_util(BytesIO(image_bytes1), BytesIO(image_bytes2))
    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")
    finally:
        if firebase_filename1:
            await remove_temp_file(firebase_filename1)
        if firebase_filename2:
            await remove_temp_file(firebase_filename2)