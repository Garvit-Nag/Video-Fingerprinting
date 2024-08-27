import os
import re
import tempfile
import shutil
import hashlib
from datetime import datetime
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock_cloud_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def save_uploaded_image(file: UploadFile) -> tuple:
    """
    Save the uploaded image to a temporary directory with a hashed name,
    and return both the original filename and the temporary file path for processing.
    """
    original_filename = file.filename
    _, file_extension = os.path.splitext(original_filename)
    
    # Create a hashed filename for processing
    file_hash = hashlib.md5(original_filename.encode()).hexdigest()
    hashed_filename = f"{file_hash}{file_extension}"
    
    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp(dir=STORAGE_DIR)
    temp_file_path = os.path.join(temp_dir, hashed_filename)
    
    logger.debug(f"Attempting to save image: {temp_file_path}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if os.path.exists(temp_file_path):
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"Image saved successfully: {temp_file_path} (Size: {file_size} bytes)")
        else:
            logger.error(f"Image not found after saving: {temp_file_path}")
            raise FileNotFoundError(f"Failed to save image: {temp_file_path}")
        
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise
    
    return original_filename, temp_file_path



def save_uploaded_file(file: UploadFile) -> tuple:
    """
    Save the uploaded file to a temporary directory with a hashed name,
    and return both the original file path and the temporary file path for processing.
    """
    original_filename = file.filename
    _, file_extension = os.path.splitext(original_filename)
    
    # Create a hashed filename for processing
    file_hash = hashlib.md5(original_filename.encode()).hexdigest()
    hashed_filename = f"{file_hash}{file_extension}"
    
    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp(dir=STORAGE_DIR)
    temp_file_path = os.path.join(temp_dir, hashed_filename)
    
    logger.debug(f"Attempting to save file: {temp_file_path}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if os.path.exists(temp_file_path):
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"File saved successfully: {temp_file_path} (Size: {file_size} bytes)")
        else:
            logger.error(f"File not found after saving: {temp_file_path}")
            raise FileNotFoundError(f"Failed to save file: {temp_file_path}")
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise
    
    return original_filename, temp_file_path
    
def sanitize_filename(filename, max_length=200):
    # Remove any non-word characters (everything except numbers and letters)
    name = re.sub(r"[^\w\s-]", "", os.path.splitext(filename)[0])
    # Replace all runs of whitespace with a single underscore
    name = re.sub(r"\s+", "_", name)
    # Truncate the name if it's too long
    name = name[:max_length]
    # Append a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Get the file extension
    _, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"