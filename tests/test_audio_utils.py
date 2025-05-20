"""
Tests for audio_utils module.
"""
import os
import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import audio_utils


def test_generate_sine_wave():
    """Test sine wave generation."""
    # Test parameters
    frequency = 440.0
    duration = 1.0
    sample_rate = 22050
    
    # Generate sine wave
    audio_data, sr = audio_utils.generate_sine_wave(
        frequency=frequency,
        duration=duration,
        sample_rate=sample_rate
    )
    
    # Check output type and shape
    assert isinstance(audio_data, np.ndarray)
    assert sr == sample_rate
    assert len(audio_data) == int(duration * sample_rate)
    
    # Check the frequency content (approximately)
    # For a pure tone, the peak in the spectrum should be at the target frequency
    frequencies, magnitudes = audio_utils.compute_fft(audio_data, sample_rate)
    peak_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC component
    peak_freq = frequencies[peak_idx]
    
    # Check if the peak frequency is close to the target (within 1%)
    assert abs(peak_freq - frequency) / frequency < 0.01


def test_compute_fft():
    """Test FFT computation."""
    # Create a simple signal with known frequency components
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal with 440 Hz and 880 Hz components
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    
    # Compute FFT
    frequencies, magnitudes = audio_utils.compute_fft(signal, sample_rate)
    
    # Find peaks
    # Exclude very low frequencies and get highest magnitude indices
    start_idx = int(100 * len(frequencies) / (sample_rate / 2))  # Start from 100 Hz
    top_indices = np.argsort(magnitudes[start_idx:])[-2:] + start_idx
    top_freqs = frequencies[top_indices]
    
    # Check if the peaks are close to 440 Hz and 880 Hz
    assert any(abs(freq - 440) / 440 < 0.05 for freq in top_freqs)
    assert any(abs(freq - 880) / 880 < 0.05 for freq in top_freqs)


def test_compute_power_db():
    """Test conversion to decibels."""
    # Test with known values
    magnitudes = np.array([1.0, 0.5, 0.1, 0.01])
    
    # Compute power in dB
    power_db = audio_utils.compute_power_db(magnitudes)
    
    # Expected values: 10*log10(mag^2) for reference power of 1.0
    expected_db = np.array([0.0, -6.02, -20.0, -40.0])
    
    # Check if values are close to expected
    assert np.allclose(power_db, expected_db, atol=0.1)
    
    # Test min threshold
    magnitudes_with_zero = np.array([1.0, 0.0])
    power_db = audio_utils.compute_power_db(magnitudes_with_zero)
    
    # The second value should be limited by amin
    assert power_db[1] < -100  # Should be a very negative dB value


def test_compute_stft():
    """Test STFT computation."""
    # Create a simple signal
    sample_rate = 22050
    duration = 1.0
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration), endpoint=False))
    
    # Compute STFT
    stft, frequencies, times = audio_utils.compute_stft(audio_data, sample_rate)
    
    # Check output shapes and types
    assert isinstance(stft, np.ndarray)
    assert isinstance(frequencies, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert stft.shape[0] == len(frequencies)
    assert stft.shape[1] == len(times)
    
    # Check frequency range
    assert frequencies[0] >= 0
    assert frequencies[-1] <= sample_rate / 2 