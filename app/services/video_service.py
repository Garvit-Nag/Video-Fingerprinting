import cv2
import numpy as np
from scipy.fftpack import dct
import imagehash
from PIL import Image
import logging
from app.utils.hash_utils import compute_video_hash, compute_frame_hashes
from app.services.audio_service import extract_audio_features, compute_audio_hash, compute_audio_hashes

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

def fingerprint_video(video_path):
    logging.info("Fingerprinting video: %s", video_path)
    video_features = extract_video_features(video_path)
    audio_features = extract_audio_features(video_path)
    
    video_hash = compute_video_hash(video_features)
    audio_hash = compute_audio_hash(audio_features)
    frame_hashes = compute_frame_hashes(video_path)
    audio_hashes = compute_audio_hashes(video_path)
    
    collective_audio_hash = compute_audio_hash(audio_features)
    
    logging.info("Finished fingerprinting video.")
    
    return {
        'frame_hashes': frame_hashes,
        'audio_hashes': audio_hashes,
        'robust_audio_hash': str(collective_audio_hash) if collective_audio_hash else None,
        'robust_video_hash': str(video_hash),
    }

def compare_videos(video_path1, video_path2):
    fp1 = fingerprint_video(video_path1)
    fp2 = fingerprint_video(video_path2)
    
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