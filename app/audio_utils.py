"""
Audio utility functions for loading, recording, and basic signal processing.
"""
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    audio_dir = Path("audio_inputs")
    audio_dir.mkdir(exist_ok=True)


def load_audio_file(file_path: Union[str, Path], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path with optional resampling.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate. If None, uses the file's native sample rate.
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return audio_data, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def save_audio_from_upload(uploaded_file, save_path: Optional[str] = None) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        save_path: Optional custom save path
        
    Returns:
        Path where the file was saved
    """
    if save_path is None:
        filename = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        save_path = os.path.join("audio_inputs", filename)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the data to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path


def record_audio(duration: float = 3.0, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    print(f"Recording {duration} seconds of audio...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait for recording to finish
    
    # Convert to mono and correct shape
    audio_data = audio_data.flatten()
    
    return audio_data, sample_rate


def save_recording(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Save recorded audio to file.
    
    Args:
        audio_data: Numpy array of audio data
        sample_rate: Sample rate of the audio
        
    Returns:
        Path where the recording was saved
    """
    filename = f"recording_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    save_path = os.path.join("audio_inputs", filename)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the data to disk
    sf.write(save_path, audio_data, sample_rate)
    
    return save_path


def generate_sine_wave(frequency: float = 440.0, 
                      duration: float = 2.0,
                      sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Generate a sine wave at the specified frequency.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Apply fade-in and fade-out to avoid clicks
    fade_duration = 0.01  # 10ms fade
    fade_samples = int(fade_duration * sample_rate)
    
    if fade_samples * 2 < len(audio_data):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio_data[:fade_samples] *= fade_in
        audio_data[-fade_samples:] *= fade_out
    
    return audio_data, sample_rate


def compute_fft(audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the real FFT of an audio signal.
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (frequencies, magnitudes)
    """
    # Apply Hann window
    windowed_data = audio_data * np.hanning(len(audio_data))
    
    # Compute FFT
    fft_data = np.fft.rfft(windowed_data)
    
    # Compute magnitude spectrum (linear)
    magnitudes = np.abs(fft_data)
    
    # Compute frequency bins
    frequencies = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    
    return frequencies, magnitudes


def compute_power_db(magnitudes: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Convert linear magnitudes to decibels.
    
    Args:
        magnitudes: FFT magnitudes
        ref: Reference value for dB conversion
        amin: Minimum threshold (to avoid log(0))
        
    Returns:
        Magnitude spectrum in dB
    """
    # Calculate power
    power = (magnitudes ** 2)
    
    # Convert to dB with safeguards against numerical issues
    power_db = 10.0 * np.log10(np.maximum(amin, power) / ref)
    
    return power_db


def compute_stft(audio_data: np.ndarray, 
                sample_rate: int,
                n_fft: int = 2048,
                hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT).
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Number of samples between frames
        
    Returns:
        Tuple of (stft_matrix, frequencies, times)
    """
    # Compute STFT
    stft_matrix = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    
    # Compute magnitude spectrum
    magnitudes = np.abs(stft_matrix)
    
    # Convert to dB
    log_spectrogram = librosa.amplitude_to_db(magnitudes, ref=np.max)
    
    # Get frequency and time axes
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.times_like(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    
    return log_spectrogram, frequencies, times


def compute_mel_spectrogram(audio_data: np.ndarray,
                           sample_rate: int,
                           n_fft: int = 2048,
                           hop_length: int = 512,
                           n_mels: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Mel spectrogram.
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mels: Number of Mel bands
        
    Returns:
        Tuple of (mel_spectrogram, mel_frequencies, times)
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Get mel frequency and time axes
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sample_rate/2)
    times = librosa.times_like(mel_spec, sr=sample_rate, hop_length=hop_length)
    
    return log_mel_spec, mel_frequencies, times


def compute_mfcc(audio_data: np.ndarray,
                sample_rate: int,
                n_mfcc: int = 20,
                n_fft: int = 2048,
                hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mel-frequency cepstral coefficients (MFCCs).
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCCs to return
        n_fft: FFT window size
        hop_length: Number of samples between frames
        
    Returns:
        Tuple of (mfcc_features, times)
    """
    # Compute MFCCs
    mfcc_features = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Get time axis
    times = librosa.times_like(mfcc_features, sr=sample_rate, hop_length=hop_length)
    
    return mfcc_features, times 