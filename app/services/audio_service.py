import numpy as np
import librosa
import imagehash
from PIL import Image
import logging
import subprocess
import os
from pydub import AudioSegment

def extract_audio_features(video_path):
    logging.info(f"Extracting audio features from: {video_path}")
    try:
        # Use ffmpeg to extract audio to a temporary wav file
        temp_audio_path = video_path + ".wav"
        subprocess.run(["ffmpeg", "-i", video_path, "-acodec", "pcm_s16le", "-ar", "44100", temp_audio_path], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load the audio file using librosa
        y, sr = librosa.load(temp_audio_path, sr=None)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Clean up temporary audio file
        os.remove(temp_audio_path)
        
        return mfcc
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error extracting audio features: {str(e)}")
        raise

def compute_audio_hash(features):
    logging.info("Computing audio hash.")
    if features is None:
        return None
    # Ensure the features are properly shaped for hashing
    features_2d = features.reshape(features.shape[0], -1)
    features_2d = (features_2d - np.min(features_2d)) / (np.max(features_2d) - np.min(features_2d))
    features_2d = (features_2d * 255).astype(np.uint8)
    return imagehash.phash(Image.fromarray(features_2d))

def compute_audio_hashes(video_path):
    logging.info("Computing audio hashes from: %s", video_path)
    audio = AudioSegment.from_file(video_path)
    samples = np.array(audio.get_array_of_samples())
    mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=audio.frame_rate, n_mfcc=13)
    audio_hashes = []
    for mfcc in mfccs.T:
        audio_hash = imagehash.average_hash(Image.fromarray(mfcc.reshape(13, 1)))
        audio_hashes.append(str(audio_hash))
    logging.info("Finished computing audio hashes.")
    return audio_hashes