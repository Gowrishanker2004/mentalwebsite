import numpy as np
import librosa
import soundfile as sf

def extract_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=22050)

    # Extract 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or trim MFCCs to 40x40
    if mfcc.shape[1] < 40:
        pad_width = 40 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :40]

    # Reshape for model input
    mfcc = mfcc.reshape(1, 40, 40, 1)

    return mfcc
