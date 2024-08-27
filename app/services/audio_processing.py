import numpy as np
import librosa
import hashlib
from app.utils.audio_analysis import audio_spectral_analysis

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
    