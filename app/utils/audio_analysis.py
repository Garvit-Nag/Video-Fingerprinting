import numpy as np
import librosa
from fastapi import UploadFile
import io

def audio_spectral_analysis(audio_path: str) -> np.ndarray:
    # Load the audio file using librosa
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    return mfccs