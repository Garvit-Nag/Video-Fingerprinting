import cv2
import numpy as np
from scipy.fftpack import dct
import imagehash
from PIL import Image
import logging
from app.utils.hash_utils import compute_video_hash, compute_frame_hashes
from app.services.audio_service import extract_audio_features, compute_audio_hash, compute_audio_hashes
from app.utils.file_utils import download_file, remove_temp_file
from app.core.firebase_config import firebase_bucket
import os

def extract_video_features(video_path):
    logging.info("Extracting video features from: %s", video_path)
    cap = cv2.VideoCapture(video_path)
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        dct_frame = dct(dct(resized.T, norm='ortho').T, norm='ortho')
        features.append(dct_frame[:8, :8].flatten())
    cap.release()
    logging.info("Finished extracting video features.")
    return np.array(features)

async def fingerprint_video(video_url):
    logging.info("Fingerprinting video: %s", video_url)
    firebase_filename = None
    try:
        firebase_filename = await download_file(video_url)
        
        # Download the file from Firebase
        blob = firebase_bucket.blob(firebase_filename)
        video_bytes = blob.download_as_bytes()
        
        # Use cv2.VideoCapture with a memory buffer
        video_array = np.asarray(bytearray(video_bytes), dtype=np.uint8)
        cap = cv2.VideoCapture()
        cap.open(cv2.CAP_OPENCV_MJPEG, memoryBuf=video_array.tobytes())
        
        video_features = extract_video_features(cap)
        audio_features = extract_audio_features(video_bytes)
        
        video_hash = compute_video_hash(video_features)
        audio_hashes = compute_audio_hash(audio_features)
        frame_hashes = compute_frame_hashes(cap)
        audio_hashes = compute_audio_hashes(video_bytes)
        
        collective_audio_hash = compute_audio_hash(audio_features)

        logging.info("Finished fingerprinting video.")

        return {
            'frame_hashes': frame_hashes,
            'audio_hashes': audio_hashes,
            'robust_audio_hash': str(collective_audio_hash) if collective_audio_hash else None,
            'robust_video_hash': str(video_hash),
        }
    finally:
        if firebase_filename:
            await remove_temp_file(firebase_filename)
        if 'cap' in locals():
            cap.release()

async def compare_videos(video_url1, video_url2):
    fp1 = await fingerprint_video(video_url1)
    fp2 = await fingerprint_video(video_url2)

    video_similarity = 1 - (imagehash.hex_to_hash(fp1['robust_video_hash']) - imagehash.hex_to_hash(fp2['robust_video_hash'])) / 64.0
    audio_similarity = 1 - (imagehash.hex_to_hash(fp1['robust_audio_hash']) - imagehash.hex_to_hash(fp2['robust_audio_hash'])) / 64.0

    overall_similarity = (video_similarity + audio_similarity) / 2
    is_same_content = overall_similarity > 0.9  # You can adjust this threshold

    logging.info("Comparison result - Video Similarity: %f, Audio Similarity: %f, Overall Similarity: %f, Is Same Content: %s",
                 video_similarity, audio_similarity, overall_similarity, is_same_content)

    return {
        "video_similarity": video_similarity,
        "audio_similarity": audio_similarity,
        "overall_similarity": overall_similarity,
        "is_same_content": is_same_content
    }