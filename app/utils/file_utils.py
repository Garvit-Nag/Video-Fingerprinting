import os
import aiohttp
import uuid 
import logging
from urllib.parse import urlparse
from app.core.firebase_config import firebase_bucket

async def download_file(url: str) -> str:
    # Extract the file extension from the URL
    parsed_url = urlparse(url)
    file_extension = os.path.splitext(parsed_url.path)[1]

    # If no extension is found, default to .tmp
    if not file_extension:
        file_extension = '.tmp'

    # Generate a unique filename
    filename = f"{uuid.uuid4()}{file_extension}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download file: HTTP {response.status}")
                
                # Upload the file content to Firebase
                blob = firebase_bucket.blob(filename)
                blob.upload_from_string(await response.read(), content_type=response.headers.get('content-type'))

        logging.info(f"File downloaded and saved to Firebase: {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        raise

async def remove_temp_file(filename: str):
    try:
        blob = firebase_bucket.blob(filename)
        blob.delete()
        logging.info(f"Temporary file deleted from Firebase: {filename}")
    except Exception as e:
        logging.error(f"Error deleting temporary file from Firebase: {str(e)}")