import os
import re
import tempfile
import librosa
import hashlib
from fastapi import FastAPI, UploadFile, HTTPException
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

# Supported file formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']


# Load environment variables from .env file
load_dotenv()

def get_safe_filename(filename):
    # Create a hash of the original filename
    hash_object = hashlib.md5(filename.encode())
    file_hash = hash_object.hexdigest()
    # Get the file extension
    _, ext = os.path.splitext(filename)
    # Return a combination of the hash and the original extension
    return f"{file_hash}{ext}"

def sanitize_filename(filename):
    # Get a safe file name
    filename = get_safe_filename(filename)
    # Remove any non-word characters (everything except numbers and letters)
    filename = re.sub(r"[^\w\s-]", "", filename)
    # Replace all runs of whitespace with a single underscore
    filename = re.sub(r"\s+", "_", filename)
    return filename

app = FastAPI()

@app.post("/verify_video")
async def verify_video(video_file: UploadFile):
    # Check if the file format is supported
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format. Supported formats are: {', '.join(SUPPORTED_VIDEO_FORMATS)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Sanitize the filename
        safe_filename = sanitize_filename(video_file.filename)
        video_path = os.path.join(temp_dir, safe_filename)
        
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
        collective_audio_hash = None

        if has_audio:
            # Extract audio from the video and save it
            audio_path = os.path.join(temp_dir, "audio.wav")
            extract_audio_from_video(video_path, audio_path)

            # Perform audio spectral analysis
            audio_hash = audio_spectral_analysis(audio_path)
            audio_hash = audio_hash.tolist()
            
            # Compute collective audio hash
            collective_audio_hash = compute_collective_audio_hash(audio_path)

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
        "collective_audio_hash": collective_audio_hash,
        "video_hash": video_hash
    }

@app.post("/verify_video_only")
async def verify_video_only(video_file: UploadFile):
    # Check if the file format is supported
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format. Supported formats are: {', '.join(SUPPORTED_VIDEO_FORMATS)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Sanitize the filename
        safe_filename = sanitize_filename(video_file.filename)
        video_path = os.path.join(temp_dir, safe_filename)
        
        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        # Extract frames from the saved video file
        frames = extract_frames(video_path)

        # Compute the video-level hash
        video_hash = compute_video_hash(frames)

    # Return only the video hash
    return {
        "video_hash": video_hash
    }
    
@app.post("/verify_image")
async def verify_image(image_file: UploadFile):
    # Check if the file format is supported
    file_ext = os.path.splitext(image_file.filename)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported formats are: {', '.join(SUPPORTED_IMAGE_FORMATS)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Sanitize the filename
        safe_filename = sanitize_filename(image_file.filename)
        image_path = os.path.join(temp_dir, safe_filename)
        
        with open(image_path, "wb") as f:
            f.write(await image_file.read())

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Unable to process the image")

        # Generate perceptual hash for the image
        image_hash = perceptual_hash(image)

    # Return the image hash
    return {
        "image_hash": image_hash
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
        
def compute_collective_frame_hash(frame_hashes: List[str]) -> str:
    # Concatenate all frame hashes
    combined_hash = ''.join(frame_hashes)
    
    # Compute SHA-256 hash of the combined string
    return hashlib.sha256(combined_hash.encode()).hexdigest()

def compute_collective_audio_hash(audio_path: str, segment_duration: float = 1.0) -> str:
    # Load the audio file using librosa (handled inside audio_spectral_analysis)
    y, sr = librosa.load(audio_path, sr=None)
    
    segment_hashes = []
    
    # Compute hash for each segment of the audio
    for start_time in np.arange(0, len(y), int(segment_duration * sr)):
        end_time = min(start_time + int(segment_duration * sr), len(y))
        segment = y[start_time:end_time]
        
        # Perform spectral analysis on the segment
        segment_mfcc = librosa.feature.mfcc(y=segment, sr=sr)
        
        # Convert the MFCCs to bytes and compute the hash for the segment
        segment_mfcc_bytes = segment_mfcc.tobytes()
        segment_hash = hashlib.sha256(segment_mfcc_bytes).hexdigest()
        segment_hashes.append(segment_hash)
    
    # Concatenate all segment hashes
    combined_hash = ''.join(segment_hashes)
    
    # Compute SHA-256 hash of the combined string
    return hashlib.sha256(combined_hash.encode()).hexdigest()