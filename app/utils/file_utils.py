import os
import aiohttp
import aiofiles
from app.core.config import TEMP_STORAGE
import logging
from urllib.parse import urlparse

async def download_file(url: str) -> str:
    # Extract the file extension from the URL
    parsed_url = urlparse(url)
    file_extension = os.path.splitext(parsed_url.path)[1]
    
    # If no extension is found, default to .tmp
    if not file_extension:
        file_extension = '.tmp'
    
    temp_path = os.path.join(TEMP_STORAGE, f"temp_{os.urandom(8).hex()}{file_extension}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download file: HTTP {response.status}")
                async with aiofiles.open(temp_path, mode='wb') as f:
                    await f.write(await response.read())
        logging.info(f"File downloaded and saved to: {temp_path}")
        return temp_path
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        raise

async def remove_temp_file(temp_path: str):
    try:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logging.info(f"Temporary file deleted: {temp_path}")
    except Exception as e:
        logging.error(f"Error deleting temporary file: {str(e)}")