import cv2
import os
import ffmpeg
import moviepy.editor as mp
from typing import List
import numpy as np
from app.utils.perceptual_hashing import compute_video_hash
import logging
import hashlib
import librosa

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

async def extract_frames(video_path: str, max_frames: int = 100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                frames.append(frame)
            frame_count += 1
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
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

def is_valid_video_file(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream is not None
    except ffmpeg.Error:
        return False
    
async def compute_video_hash_only(video_path: str):
    try:
        if not is_valid_video_file(video_path):
            raise ValueError(f"Invalid or corrupted video file: {video_path}")
        
        frames = await extract_frames(video_path)
        video_hash = compute_video_hash(frames)
        return video_hash
    except Exception as e:
        logger.error(f"Error computing video hash: {str(e)}", exc_info=True)
        raise

def verify_video_format(filename: str):
    file_ext = os.path.splitext(filename)[1].lower()
    if not file_ext:
        raise ValueError("File must have an extension")
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format. Supported formats are: {', '.join(SUPPORTED_VIDEO_FORMATS)}")