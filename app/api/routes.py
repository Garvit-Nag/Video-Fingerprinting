from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.video_service import fingerprint_video, compare_videos
from app.services.image_service import verify_image, compare_images
from app.utils.file_utils import download_file, remove_temp_file
import logging

router = APIRouter()

class ContentRequest(BaseModel):
    url: str
    
class CompareRequest(BaseModel):
    url1: str
    url2: str


@router.post("/fingerprint")
async def create_fingerprint(request: ContentRequest):
    logging.info("Received request to create fingerprint.")
    temp_path = None
    try:
        temp_path = await download_file(request.url)
        fingerprint = fingerprint_video(temp_path)
        return fingerprint
    except Exception as e:
        logging.error(f"Error creating fingerprint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path:
            await remove_temp_file(temp_path)

@router.post("/compare_videos")
async def compare_videos_route(request: CompareRequest):
    logging.info("Received request to compare videos.")
    temp_paths = []
    try:
        temp_path1 = await download_file(request.url1)
        temp_path2 = await download_file(request.url2)
        temp_paths = [temp_path1, temp_path2]
        
        result = compare_videos(temp_path1, temp_path2)
        return result
    except Exception as e:
        logging.error(f"Error comparing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        for path in temp_paths:
            await remove_temp_file(path)

@router.post("/verify_image")
async def verify_image_route(request: ContentRequest):
    temp_path = None
    try:
        temp_path = await download_file(request.url)
        image_hash = verify_image(temp_path)
        return {"image_hash": image_hash}
    except Exception as e:
        logging.error(f"Error verifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if temp_path:
            await remove_temp_file(temp_path)

@router.post("/compare_images")
async def compare_images_route(request: CompareRequest):
    logging.info("Received request to compare images.")
    temp_paths = []
    try:
        temp_path1 = await download_file(request.url1)
        temp_path2 = await download_file(request.url2)
        temp_paths = [temp_path1, temp_path2]

        result = compare_images(temp_path1, temp_path2)
        return result
    except Exception as e:
        logging.error(f"Error comparing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")
    finally:
        for path in temp_paths:
            await remove_temp_file(path)