from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.video_service import fingerprint_video, compare_videos
from app.services.image_service import verify_image, compare_images
from app.utils.file_utils import download_file, remove_temp_file

from app.services.antispoof_service import antispoof_service
from fastapi.responses import JSONResponse

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
        fingerprint = await fingerprint_video(temp_path)
        return fingerprint
    except Exception as e:
        logging.error(f"Error creating fingerprint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path:
            await remove_temp_file(temp_path)
            
@router.post("/verify_video_only")
async def verify_video_only(request: ContentRequest):
    temp_path = None
    try:
        logging.info("Received request to verify video.")
        temp_path = await download_file(request.url)
        fingerprint = await fingerprint_video(temp_path)
        # Extract robust_video_hash from the fingerprint result
        robust_video_hash = fingerprint.get('robust_video_hash')
        if not robust_video_hash:
            raise HTTPException(status_code=404, detail="Robust video hash not found.")
        return {"robust_video_hash": robust_video_hash}
    except Exception as e:
        logging.error(f"Error verifying video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path:
            await remove_temp_file(temp_path)
            
@router.post("/verify_liveness")
async def verify_liveness(request: ContentRequest):
    logging.info("Received request to verify image liveness.")
    try:
        result = await antispoof_service.verify_image(request.url)
        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error verifying image liveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/compare_videos")
async def compare_videos_route(request: CompareRequest):
    logging.info("Received request to compare videos.")
    temp_paths = []
    try:
        temp_path1 = await download_file(request.url1)
        temp_path2 = await download_file(request.url2)
        temp_paths = [temp_path1, temp_path2]
        
        result = await compare_videos(temp_path1, temp_path2)
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
        image_hash = await verify_image(temp_path)
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

        result = await compare_images(temp_path1, temp_path2)
        return result
    except Exception as e:
        logging.error(f"Error comparing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")
    finally:
        for path in temp_paths:
            await remove_temp_file(path)