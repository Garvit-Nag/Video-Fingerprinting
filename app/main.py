import os
import tempfile
from fastapi import FastAPI, UploadFile
from typing import List
import numpy as np
import cv2
import moviepy.editor as mp
from app.utils.perceptual_hashing import perceptual_hash
from app.utils.audio_analysis import audio_spectral_analysis
from app.utils.video_analysis import compute_video_hash
from app.cloud_storage_client import upload_to_cloud
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.post("/verify_video")
async def verify_video(video_file: UploadFile):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded video file to the temporary directory
        video_path = os.path.join(temp_dir, video_file.filename)
        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        # Check if the video has an audio track
        has_audio = check_audio_in_video(video_path)

        # Extract frames from the saved video file
        frames = extract_frames(video_path)

        # Generate perceptual hashes for the frames
        frame_hashes = [perceptual_hash(frame) for frame in frames]

        # Initialize audio hash
        audio_hash = None

        if has_audio:
            # Extract audio from the video and save it
            audio_path = os.path.join(temp_dir, "audio.wav")
            extract_audio_from_video(video_path, audio_path)

            # Perform audio spectral analysis
            audio_hash = audio_spectral_analysis(audio_path)
            audio_hash = audio_hash.tolist()

        # Compute the video-level hash
        video_hash = compute_video_hash(frames)

        # Upload to cloud storage if running in production
        if os.getenv('ENV') == 'production':
            video_url = await upload_to_cloud(video_file)
            # Handle the uploaded video_url if needed

        # Return the hashes
        return {
            "frame_hashes": frame_hashes,
            "audio_hash": audio_hash,
            "video_hash": video_hash
        }

def extract_frames(video_path: str) -> List[np.ndarray]:
    # Use OpenCV to extract frames from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video.")
        return frames
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        logging.warning("No frames extracted.")
        
    logging.info(f"Extracted {len(frames)} frames from video.")
    return frames

def check_audio_in_video(video_path: str) -> bool:
    # Use moviepy to check if the video has an audio track
    video = mp.VideoFileClip(video_path)
    return video.audio is not None

def extract_audio_from_video(video_path: str, audio_path: str):
    # Use moviepy to extract audio from video
    video = mp.VideoFileClip(video_path)
    if video.audio:
        video.audio.write_audiofile(audio_path)
