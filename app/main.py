import os
import asyncio
import hashlib
from typing import List
import shutil
from fastapi import FastAPI, UploadFile, HTTPException, File
from app.services.file_handling import save_uploaded_file, save_uploaded_image
from app.services.video_processing import (
    verify_video_format, is_valid_video_file, check_audio_in_video, 
    extract_frames, extract_audio_from_video, compute_video_hash_only
)
from app.services.audio_processing import compute_collective_audio_hash
from app.utils.perceptual_hashing import perceptual_hash, compute_video_hash
from app.utils.audio_analysis import audio_spectral_analysis
from app.services.image_processing import verify_image_format, process_image, compare_images, cleanup_temp_dirs
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.post("/verify_video")
async def verify_video(video_file: UploadFile = File(...)):
    logger.info(f"Received file: {video_file.filename}")
    
    # Check if the file format is supported
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if not file_ext:
        raise HTTPException(status_code=400, detail="File must have an extension")

    temp_dir = None
    try:
        # Save the uploaded file and get both original filename and temporary path
        original_filename, temp_file_path = save_uploaded_file(video_file)
        temp_dir = os.path.dirname(temp_file_path)
        
        logger.info(f"File saved. Original filename: {original_filename}")
        logger.info(f"Proceeding with video verification using: {temp_file_path}")
        
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Saved file not found: {temp_file_path}")
        
        # Check if the video file is valid
        if not is_valid_video_file(temp_file_path):
            raise ValueError(f"Invalid or corrupted video file: {temp_file_path}")
        
        # Check if the video has an audio track
        has_audio = check_audio_in_video(temp_file_path)

        # Extract frames from the saved video file
        frames = await extract_frames(temp_file_path)

        # Generate perceptual hashes for the frames
        frame_hashes = [perceptual_hash(frame) for frame in frames]

        # Initialize audio hash
        audio_hash = None
        collective_audio_hash = None

        if has_audio:
            # Extract audio from the video and save it
            audio_filename = f"audio_{os.path.splitext(os.path.basename(temp_file_path))[0]}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            extract_audio_from_video(temp_file_path, audio_path)

            # Perform audio spectral analysis
            audio_hash = audio_spectral_analysis(audio_path)
            audio_hash = audio_hash.tolist()
            
            # Compute collective audio hash
            collective_audio_hash = compute_collective_audio_hash(audio_path)

        # Compute the video-level hash
        video_hash = compute_video_hash(frames)

        # Return the hashes and file information
        return {
            "frame_hashes": frame_hashes,
            "audio_hash": audio_hash,
            "collective_audio_hash": collective_audio_hash,
            "video_hash": video_hash
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/verify_video_only")
async def verify_video_only(video_file: UploadFile = File(...)):
    logger.info(f"Received file for video-only verification: {video_file.filename}")
    
    temp_dir = None
    try:
        verify_video_format(video_file.filename)
        
        original_filename, temp_file_path = save_uploaded_file(video_file)
        temp_dir = os.path.dirname(temp_file_path)
        
        logger.info(f"File saved. Original filename: {original_filename}")
        logger.info(f"Proceeding with video-only verification using: {temp_file_path}")
        
        video_hash = await compute_video_hash_only(temp_file_path)
        
        return {"video_hash": video_hash}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing video for video-only verification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    


@app.post("/verify_image")
async def verify_image(image_file: UploadFile = File(...)):
    verify_image_format(image_file.filename)
    temp_dir = None
    try:
        original_filename, temp_file_path = save_uploaded_image(image_file)
        temp_dir = os.path.dirname(temp_file_path)
        
        logger.info(f"Image saved. Original filename: {original_filename}")
        logger.info(f"Proceeding with image verification using: {temp_file_path}")
        
        image_hash = process_image(temp_file_path)
        
        return {"image_hash": image_hash}
    finally:
        cleanup_temp_dirs([temp_dir])

@app.post("/compare_images")
async def compare_images_endpoint(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    verify_image_format(image1.filename)
    verify_image_format(image2.filename)
    temp_dirs = []
    try:
        original_filename1, temp_file_path1 = save_uploaded_image(image1)
        original_filename2, temp_file_path2 = save_uploaded_image(image2)
        
        temp_dirs.extend([os.path.dirname(temp_file_path1), os.path.dirname(temp_file_path2)])
        
        result = compare_images(temp_file_path1, temp_file_path2)
        
        return result
    finally:
        cleanup_temp_dirs(temp_dirs)
    
def compute_collective_frame_hash(frame_hashes: List[str]) -> str:
    # Concatenate all frame hashes
    combined_hash = ''.join(frame_hashes)
    
    # Compute SHA-256 hash of the combined string
    return hashlib.sha256(combined_hash.encode()).hexdigest()


