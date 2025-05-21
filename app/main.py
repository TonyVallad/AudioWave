"""
AudioWave: An interactive audio exploration and analysis tool.
"""
import os
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Literal

import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import plotly.graph_objects as go

# Import custom modules
import audio_utils
import dsp_plots
import ml

# Apply custom styling
def apply_custom_styling():
    """Apply custom CSS styling to the app."""
    st.markdown("""
        <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #2E2E2E;
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Sidebar text color adjustments for better visibility on dark bg */
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFFFFF;
        }
        
        /* Sliders - progress part (filled) */
        .stSlider [data-baseweb="slider"] div div div div {
            background-color: transparent !important;
        }
        
        /* Sliders - thumb */
        .stSlider [data-baseweb="slider"] span div {
            background-color: #00FFFF !important;
        }
        
        /* Checkboxes */
        .stCheckbox > label > div[data-testid="stMarkdownContainer"] + div div[role="presentation"] {
            background-color: #00CCAA !important;
            border-color: #00CCAA !important;
        }
        
        /* Radio buttons */
        .stRadio > label > div[data-testid="stMarkdownContainer"] + div div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] + div > div {
            background-color: transparent !important;
            border-color: #FFFFFF !important;
        }
        
        /* Button colors */
        .stButton button {
            color: white;
            background-color: #00AA99;
            border: none;
        }
        
        .stButton button:hover {
            background-color: #00CCAA;
        }
        
        /* Sidebar button colors - ensure visibility */
        [data-testid="stSidebar"] .stButton button {
            background-color: #00CCAA;
            color: #000000;
            font-weight: bold;
        }
        
        /* Tab styling - Active tab in green */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #00FF00 !important;
        }
        
        /* Tab styling - Active tab indicator */
        .stTabs [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] + div {
            background-color: #00FF00 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="AudioWave",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
apply_custom_styling()

# Constants
SAMPLE_RATE = 22050  # Default sample rate for processing
MAX_UPLOAD_SIZE_MB = 50  # Maximum upload file size in MB
DEFAULT_DURATION = 3.0  # Default recording duration in seconds

# Audio directory constants
RECORDINGS_DIR = Path("audio_inputs/recordings")
UPLOADS_DIR = Path("audio_inputs/uploads")
SYNTH_DIR = Path("audio_inputs/synthesized")
ALL_AUDIO_DIRS = [RECORDINGS_DIR, UPLOADS_DIR, SYNTH_DIR]

# Create necessary directories
def create_directories() -> None:
    """Create the necessary directories for audio files if they don't exist."""
    for directory in ALL_AUDIO_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

# Organize existing audio files into the proper directories
def organize_audio_files() -> None:
    """
    Organize existing audio files into the appropriate directories based on filename.
    Only moves files from the root audio_inputs directory.
    """
    audio_inputs = Path("audio_inputs")
    if not audio_inputs.exists():
        return
    
    # Get all files in the root audio_inputs directory
    audio_files = [f for f in audio_inputs.glob("*.wav") if f.is_file()]
    
    for file_path in audio_files:
        filename = file_path.name
        
        # Determine the appropriate directory
        if filename.startswith("recording_"):
            target_dir = RECORDINGS_DIR
        elif filename.startswith("sine_"):
            target_dir = SYNTH_DIR
        else:
            target_dir = UPLOADS_DIR
        
        # Create target path and move the file
        target_path = target_dir / filename
        try:
            shutil.move(str(file_path), str(target_path))
            print(f"Moved {filename} to {target_dir}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

# Ensure the audio directories exist
create_directories()


# Cache helper functions with st.cache_data
@st.cache_data
def load_audio_cached(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Cached version of audio loading function."""
    return audio_utils.load_audio_file(file_path, sr)


@st.cache_data
def compute_features_cached(audio_data: np.ndarray, sample_rate: int, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute and cache all audio features for a given signal.
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        file_path: Optional source file path (used as additional cache key)
        
    Returns:
        Dictionary with all computed features
    """
    # FFT and power spectrum
    frequencies, magnitudes = audio_utils.compute_fft(audio_data, sample_rate)
    power_db = audio_utils.compute_power_db(magnitudes)
    
    # STFT
    stft, stft_freqs, stft_times = audio_utils.compute_stft(audio_data, sample_rate)
    
    # Mel spectrogram
    mel_spec, mel_freqs, mel_times = audio_utils.compute_mel_spectrogram(audio_data, sample_rate)
    
    # MFCCs
    mfcc, mfcc_times = audio_utils.compute_mfcc(audio_data, sample_rate)
    
    # Return all features
    return {
        "fft": {
            "frequencies": frequencies,
            "magnitudes": magnitudes,
            "power_db": power_db
        },
        "stft": {
            "spectrogram": stft,
            "frequencies": stft_freqs,
            "times": stft_times
        },
        "mel_spec": {
            "spectrogram": mel_spec,
            "frequencies": mel_freqs,
            "times": mel_times
        },
        "mfcc": {
            "features": mfcc,
            "times": mfcc_times
        }
    }


def save_audio_from_upload(uploaded_file) -> str:
    """
    Save an uploaded file to the uploads directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path where the file was saved
    """
    filename = f"upload_{int(time.time())}_{uploaded_file.name}"
    save_path = os.path.join(UPLOADS_DIR, filename)
    
    # Save the data to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path


def save_recording(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Save recorded audio to file in the recordings directory.
    
    Args:
        audio_data: Numpy array of audio data
        sample_rate: Sample rate of the audio
        
    Returns:
        Path where the recording was saved
    """
    filename = f"recording_{int(time.time())}.wav"
    save_path = os.path.join(RECORDINGS_DIR, filename)
    
    # Save the data to disk
    sf.write(save_path, audio_data, sample_rate)
    
    return save_path


def save_synthesized_audio(audio_data: np.ndarray, sample_rate: int, frequency: float) -> str:
    """
    Save synthesized audio to file in the synthesized audio directory.
    
    Args:
        audio_data: Numpy array of audio data
        sample_rate: Sample rate of the audio
        frequency: Frequency of the generated sine wave
        
    Returns:
        Path where the file was saved
    """
    filename = f"sine_{frequency}Hz_{int(time.time())}.wav"
    save_path = os.path.join(SYNTH_DIR, filename)
    
    # Save the data to disk
    sf.write(save_path, audio_data, sample_rate)
    
    return save_path


def display_audio_info(audio_data: np.ndarray, sample_rate: int) -> None:
    """Display basic information about an audio signal."""
    duration = len(audio_data) / sample_rate
    
    # Create columns for audio information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duration", f"{duration:.2f} s")
    
    with col2:
        st.metric("Sample Rate", f"{sample_rate} Hz")
    
    with col3:
        st.metric("Samples", f"{len(audio_data)}")


def sidebar_audio_input() -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Create sidebar UI for audio input (upload or record).
    
    Returns:
        Tuple of (audio_data, sample_rate, source_path)
    """
    # Initialize session state for audio data if not present
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "sample_rate" not in st.session_state:
        st.session_state.sample_rate = None
    if "source_path" not in st.session_state:
        st.session_state.source_path = None
        
    st.sidebar.title("Audio Input")
    
    # Choose input method
    input_method = st.sidebar.radio(
        "Choose input method:",
        ("Upload WAV file", "Record from microphone")
    )
    
    audio_data = st.session_state.audio_data
    sample_rate = st.session_state.sample_rate
    source_path = st.session_state.source_path
    
    # File uploader
    if input_method == "Upload WAV file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a WAV file",
            type=["wav"],
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            try:
                # Save the uploaded file
                source_path = save_audio_from_upload(uploaded_file)
                
                # Load the audio file
                audio_data, sample_rate = load_audio_cached(source_path)
                
                # Store in session state
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.session_state.source_path = source_path
                
                # Display success message
                st.sidebar.success(f"File uploaded and loaded successfully!")
                
                # Show audio player
                st.sidebar.audio(source_path)
                
            except Exception as e:
                st.sidebar.error(f"Error loading audio file: {e}")
    
    # Microphone recorder
    else:
        st.sidebar.subheader("Microphone Recording")
        
        # Recording duration
        duration = st.sidebar.slider(
            "Recording duration (seconds):",
            min_value=1.0,
            max_value=10.0,
            value=DEFAULT_DURATION,
            step=0.5
        )
        
        # Record button
        if st.sidebar.button("Record Audio"):
            try:
                with st.sidebar.status("Recording audio..."):
                    # Record audio
                    audio_data, sample_rate = audio_utils.record_audio(duration, SAMPLE_RATE)
                    
                    # Save recording
                    source_path = save_recording(audio_data, sample_rate)
                
                # Store in session state
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.session_state.source_path = source_path
                
                # Display success message
                st.sidebar.success("Recording completed!")
                
                # Show audio player
                st.sidebar.audio(source_path)
                
            except Exception as e:
                st.sidebar.error(f"Error recording audio: {e}")
    
    # Show current audio if it exists in session state
    if source_path is None and st.session_state.source_path is not None:
        st.sidebar.subheader("Currently loaded audio")
        st.sidebar.audio(st.session_state.source_path)
    
    return audio_data, sample_rate, source_path


def waveform_tab(audio_data: np.ndarray, sample_rate: int) -> None:
    """Create the waveform visualization tab."""
    st.header("Time-Domain Waveform")
    
    # Display audio information
    display_audio_info(audio_data, sample_rate)
    
    # Plot waveform
    waveform_fig = dsp_plots.plot_waveform(audio_data, sample_rate)
    st.plotly_chart(waveform_fig, use_container_width=True)
    
    # Get time array for x-axis values
    t = np.arange(len(audio_data)) / sample_rate
    
    # Display waveform statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Amplitude", f"{np.max(np.abs(audio_data)):.4f}")
    
    with col2:
        st.metric("RMS Level", f"{np.sqrt(np.mean(audio_data**2)):.4f}")
    
    with col3:
        st.metric("Zero Crossings", f"{np.sum(np.diff(np.signbit(audio_data)))}")


def sidebar_synth() -> None:
    """Add sinusoid synthesizer to the sidebar."""
    st.sidebar.title("Synthesize Sinusoid")
    
    # Frequency control
    frequency = st.sidebar.slider(
        "Frequency (Hz):",
        min_value=20,
        max_value=20000,
        value=440,
        step=1,
        key="synth_freq_sidebar"
    )
    
    # Duration control
    duration = st.sidebar.slider(
        "Duration (seconds):",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="synth_dur_sidebar"
    )
    
    # Generate button
    if st.sidebar.button("Generate Sine Wave"):
        # Generate sine wave
        sine_wave, sample_rate = audio_utils.generate_sine_wave(
            frequency=frequency,
            duration=duration,
            sample_rate=SAMPLE_RATE
        )
        
        # Save to file for playback in the synthesized directory
        sine_path = save_synthesized_audio(sine_wave, sample_rate, frequency)
        
        # Show audio player
        st.sidebar.audio(sine_path)
        
        # Store in session state for display in main area
        if "last_synth" not in st.session_state:
            st.session_state.last_synth = {}
        
        st.session_state.last_synth = {
            "wave": sine_wave,
            "sample_rate": sample_rate,
            "frequency": frequency,
            "path": sine_path
        }


def synth_tab() -> None:
    """Create the sinusoidal synthesis tab."""
    st.header("Synthesize a Sinusoid")
    
    # Inform user that controls are in sidebar
    st.info("The sinusoid synthesis controls are now available in the sidebar. Generate a sine wave there and it will be displayed here.")
    
    # Check if we have a generated sine wave to show
    if "last_synth" in st.session_state and st.session_state.last_synth:
        synth_data = st.session_state.last_synth
        sine_wave = synth_data["wave"]
        sample_rate = synth_data["sample_rate"]
        frequency = synth_data["frequency"]
        
        # Display waveform
        waveform_fig = dsp_plots.plot_waveform(
            sine_wave, 
            sample_rate, 
            title=f"Sine Wave at {frequency} Hz"
        )
        st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Audio player
        st.audio(synth_data["path"])
        
        # Generate time array for one cycle
        cycle_samples = int(sample_rate / frequency)
        if cycle_samples > 10:  # Ensure we have enough samples for one cycle
            t_cycle = np.linspace(0, 1/frequency, cycle_samples, endpoint=False)
            y_cycle = np.sin(2 * np.pi * frequency * t_cycle)
            
            # Plot one cycle
            cycle_fig = go.Figure()
            cycle_fig.add_trace(
                go.Scatter(
                    x=t_cycle,
                    y=y_cycle,
                    mode='lines',
                    line=dict(color='red', width=2)
                )
            )
            cycle_fig.update_layout(
                title=f"One Cycle at {frequency} Hz",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(cycle_fig, use_container_width=True)
        
        # Explanation text
        st.info(
            f"This is a pure tone at {frequency} Hz. "
            f"The waveform is generated using the equation: y(t) = sin(2π × {frequency} × t)"
        )


def spectrum_tab(audio_data: np.ndarray, sample_rate: int, features: Dict[str, Any]) -> None:
    """Create the spectrum analysis tab."""
    st.header("Frequency Spectrum")
    
    # Get FFT data from cached features
    fft_data = features["fft"]
    frequencies = fft_data["frequencies"]
    power_db = fft_data["power_db"]
    
    # Plot settings
    col1, col2 = st.columns(2)
    
    with col1:
        log_scale = st.checkbox("Log Frequency Scale", value=False)
    
    with col2:
        max_freq = st.slider(
            "Maximum Frequency (Hz):",
            min_value=100,
            max_value=sample_rate // 2,
            value=min(10000, sample_rate // 2),
            step=100
        )
    
    # Filter data by frequency range
    freq_mask = frequencies <= max_freq
    plot_freqs = frequencies[freq_mask]
    plot_power = power_db[freq_mask]
    
    # Plot spectrum
    spectrum_fig = dsp_plots.plot_spectrum(
        plot_freqs, 
        None,  # Magnitudes not needed when power_db is provided
        power_db=plot_power, 
        log_freq=log_scale
    )
    st.plotly_chart(spectrum_fig, use_container_width=True)
    
    # Basic spectrum info
    dominant_freq_idx = np.argmax(power_db[1:]) + 1  # Skip DC component
    dominant_freq = frequencies[dominant_freq_idx]
    
    st.info(
        f"The dominant frequency is approximately {dominant_freq:.1f} Hz. "
        f"This spectrum was computed using the Fast Fourier Transform (FFT) with a Hann window."
    )


def spectrogram_tab(audio_data: np.ndarray, sample_rate: int, features: Dict[str, Any]) -> None:
    """Create the spectrogram tab."""
    st.header("Short-Time Fourier Transform (STFT)")
    
    # Get STFT data from cached features
    stft_data = features["stft"]
    spectrogram = stft_data["spectrogram"]
    frequencies = stft_data["frequencies"]
    times = stft_data["times"]
    
    # Plot controls
    col1, col2 = st.columns(2)
    
    with col1:
        max_freq = st.slider(
            "Maximum Frequency (Hz):",
            min_value=100,
            max_value=sample_rate // 2,
            value=min(5000, sample_rate // 2),
            step=100,
            key="stft_max_freq"
        )
    
    # Filter data by frequency range
    freq_mask = frequencies <= max_freq
    plot_spec = spectrogram[freq_mask, :]
    plot_freqs = frequencies[freq_mask]
    
    # Plot STFT
    stft_fig = dsp_plots.plot_spectrogram(
        plot_spec,
        plot_freqs,
        times,
        title="STFT Spectrogram"
    )
    st.plotly_chart(stft_fig, use_container_width=True)
    
    # Explanation
    st.info(
        "The Short-Time Fourier Transform (STFT) shows how the frequency content of a signal "
        "changes over time. The spectrogram displays power (in dB) across frequency and time, "
        "with brighter colors representing higher energy."
    )
    
    # Show Mel spectrogram
    st.subheader("Mel Spectrogram")
    
    # Get Mel spectrogram from cached features
    mel_data = features["mel_spec"]
    mel_spec = mel_data["spectrogram"]
    mel_freqs = mel_data["frequencies"]
    mel_times = mel_data["times"]
    
    # Plot Mel spectrogram
    mel_fig = dsp_plots.plot_mel_spectrogram(
        mel_spec,
        mel_freqs,
        mel_times,
        title="Mel Spectrogram"
    )
    st.plotly_chart(mel_fig, use_container_width=True)
    
    # Explanation
    st.info(
        "The Mel spectrogram converts the frequency scale to the Mel scale, which better "
        "represents how humans perceive pitch. This is often more useful for audio analysis "
        "and is commonly used as input for machine learning models."
    )


def waterfall_tab(audio_data: np.ndarray, sample_rate: int, features: Dict[str, Any]) -> None:
    """Create the waterfall visualization tab."""
    st.header("Waterfall Visualizations")
    
    # Get STFT data from cached features
    stft_data = features["stft"]
    spectrogram = stft_data["spectrogram"]
    frequencies = stft_data["frequencies"]
    times = stft_data["times"]
    
    # Plot controls
    col1, col2 = st.columns(2)
    
    with col1:
        max_freq = st.slider(
            "Maximum Frequency (Hz):",
            min_value=100,
            max_value=sample_rate // 2,
            value=min(5000, sample_rate // 2),
            step=100,
            key="waterfall_max_freq"
        )
    
    with col2:
        visualization_type = st.radio(
            "Visualization Type:",
            ["3D Surface", "2D Cascade", "Both"],
            horizontal=True,
            key="waterfall_type"
        )
    
    # Filter data by frequency range
    freq_mask = frequencies <= max_freq
    plot_spec = spectrogram[freq_mask, :]
    plot_freqs = frequencies[freq_mask]
    
    # Show 3D waterfall visualization if selected
    if visualization_type in ["3D Surface", "Both"]:
        st.subheader("3D Waterfall Visualization")
        waterfall_fig = dsp_plots.plot_waterfall(
            plot_spec,
            plot_freqs,
            times,
            title="3D Waterfall Spectral Visualization"
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)
        
        # Explanation and controls guidance for 3D
        st.info(
            "The 3D waterfall plot shows the spectral content over time in a 3D visualization, with time flowing "
            "vertically. Colors and height represent the energy at each frequency point."
        )
        
        st.markdown("""
        **3D Interaction Tips:**
        - Click and drag to rotate the view
        - Scroll to zoom in/out
        - Double-click to reset the view
        - Right-click and drag to pan
        """)
    
    # Show 2D waterfall visualization if selected
    if visualization_type in ["2D Cascade", "Both"]:
        st.subheader("2D Waterfall Visualization")
        
        # No need for max_traces control anymore as it's not used
        waterfall_2d_fig = dsp_plots.plot_waterfall_2d(
            plot_spec,
            plot_freqs,
            times,
            title="2D Waterfall Spectrogram"
        )
        st.plotly_chart(waterfall_2d_fig, use_container_width=True)
        
        # Explanation for 2D
        st.info(
            "This 2D waterfall visualization shows frequency on the x-axis and time flowing from bottom to top on the y-axis, "
            "with color representing spectral power. This provides a clear view of how frequency content changes over time."
        )


def dashboard_tab(audio_data: np.ndarray, sample_rate: int, features: Dict[str, Any]) -> None:
    """Create a dashboard with condensed visualizations of all key features."""
    st.header("Audio Analysis Dashboard")
    
    # Display basic audio info in a compact format
    duration = len(audio_data) / sample_rate
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{duration:.2f} s")
    with col2:
        st.metric("Sample Rate", f"{sample_rate} Hz")
    with col3:
        st.metric("Samples", f"{len(audio_data):,}")
    with col4:
        st.metric("RMS Level", f"{np.sqrt(np.mean(audio_data**2)):.4f}")
    
    # Create a two-column layout for the visualizations
    left_col, right_col = st.columns(2)
    
    # Get data from features
    stft_data = features["stft"]
    spectrogram = stft_data["spectrogram"]
    frequencies = stft_data["frequencies"]
    times = stft_data["times"]
    
    fft_data = features["fft"]
    fft_freqs = fft_data["frequencies"]
    power_db = fft_data["power_db"]
    magnitudes = fft_data["magnitudes"]
    
    mel_data = features["mel_spec"]
    mel_spec = mel_data["spectrogram"]
    mel_freqs = mel_data["frequencies"]
    
    # Filter for display
    max_freq = min(5000, sample_rate // 2)
    freq_mask = frequencies <= max_freq
    fft_mask = fft_freqs <= max_freq
    
    # Left column: Waveform and Spectrum
    with left_col:
        # Waveform visualization
        st.subheader("Waveform")
        waveform_fig = dsp_plots.plot_waveform(audio_data, sample_rate)
        waveform_fig.update_layout(height=250)
        st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Log-scale Frequency Spectrum visualization
        st.subheader("Frequency Spectrum")
        spectrum_fig = dsp_plots.plot_spectrum(
            fft_freqs[fft_mask], 
            None, 
            power_db[fft_mask], 
            log_freq=True,
            title="Audio Spectrum"
        )
        spectrum_fig.update_layout(height=250)
        st.plotly_chart(spectrum_fig, use_container_width=True)
        
        # Linear-scale Frequency Spectrum
        st.subheader("Frequency spectrum")
        linear_spectrum_fig = dsp_plots.plot_spectrum(
            fft_freqs[fft_mask], 
            None, 
            power_db[fft_mask],
            log_freq=False,
            title="Frequency spectrum with linear scale"
        )
        linear_spectrum_fig.update_layout(height=250)
        st.plotly_chart(linear_spectrum_fig, use_container_width=True)
    
    # Right column: Mel Spectrogram, STFT, 2D and 3D Waterfall
    with right_col:
        # Additional Visualizations header
        st.subheader("Additional Visualizations")
        
        # Mel Spectrogram
        mel_fig = dsp_plots.plot_mel_spectrogram(
            mel_spec,
            mel_freqs,
            times,
            title="Mel Spectrogram"
        )
        mel_fig.update_layout(height=250)
        st.plotly_chart(mel_fig, use_container_width=True)
        
        # STFT Spectrogram visualization
        st.subheader("STFT Spectrogram")
        stft_fig = dsp_plots.plot_spectrogram(
            spectrogram[freq_mask, :], 
            frequencies[freq_mask], 
            times
        )
        stft_fig.update_layout(height=250)
        st.plotly_chart(stft_fig, use_container_width=True)
        
        # 2D Waterfall visualization
        st.subheader("Waterfall View")
        waterfall_2d_fig = dsp_plots.plot_waterfall_2d(
            spectrogram[freq_mask, :],
            frequencies[freq_mask],
            times,
            title="2D Waterfall Plot"
        )
        waterfall_2d_fig.update_layout(height=250)
        st.plotly_chart(waterfall_2d_fig, use_container_width=True)
    
    # 3D Waterfall visualization - Full width outside of columns
    st.subheader("Waterfall Plot")
    waterfall_fig = dsp_plots.plot_waterfall(
        spectrogram[freq_mask, :],
        frequencies[freq_mask],
        times,
        title="Waterfall Plot"
    )
    waterfall_fig.update_layout(height=700)
    st.plotly_chart(waterfall_fig, use_container_width=True)


def sample_analysis_tab() -> None:
    """Create the sample analysis tab with pre-loaded samples."""
    st.header("Audio Analysis Samples")
    
    # Sample categories and descriptions
    sample_categories = {
        "Environmental": [
            {"name": "City Street", "description": "Urban ambient noise with traffic and people"},
            {"name": "Forest", "description": "Natural ambience with birds and wind"},
            {"name": "Rain", "description": "Heavy rainfall sound"}
        ],
        "Instruments": [
            {"name": "Piano", "description": "Piano chord progression"},
            {"name": "Guitar", "description": "Acoustic guitar strumming"},
            {"name": "Drums", "description": "Drum beat pattern"}
        ]
    }
    
    # Sample selection
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category:", list(sample_categories.keys()))
    
    with col2:
        # Get samples for the selected category
        samples = sample_categories[category]
        sample_names = [sample["name"] for sample in samples]
        selected_sample = st.selectbox("Sample:", sample_names)
    
    # Find the selected sample
    selected_sample_info = next(
        (sample for sample in samples if sample["name"] == selected_sample), 
        None
    )
    
    if selected_sample_info:
        st.write(f"**Description:** {selected_sample_info['description']}")
        
        # Display notice about sample data
        st.info(
            "Note: These are placeholder descriptions. "
            "In a real application, you would load actual audio samples from the UrbanSound8K "
            "or ESC-50 datasets and perform analysis on them."
        )
        
        # Instructions for adding real samples
        st.write(
            "To use real sample data, download the UrbanSound8K dataset from: "
            "[https://urbansounddataset.weebly.com/](https://urbansounddataset.weebly.com/) "
            "or ESC-50 from: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)"
        )
        
        # Simulated analysis display
        st.subheader("Analysis for selected sample would appear here")
        st.write(
            "This tab would load and display STFT, Mel-spectrograms, and MFCCs "
            "for the selected samples. In a real implementation, you would load "
            "actual audio files and compute their features."
        )


def classification_tab() -> None:
    """Create the classification tab for audio classifier training and prediction."""
    st.header("Audio Classification")
    
    # Initialize session state for dataset
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = ml.AudioDataManager()
    
    # Tabs for dataset building and classification
    dataset_tab, train_tab, predict_tab = st.tabs(
        ["1. Build Dataset", "2. Train Classifier", "3. Predict"]
    )
    
    # Dataset building tab
    with dataset_tab:
        st.subheader("Add Audio Files to Dataset")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload a WAV file:",
            type=["wav"],
            key="clf_file_upload"
        )
        
        # Class label input
        label = st.text_input("Class Label:", placeholder="E.g., dog_bark, car_horn, etc.")
        
        # Add file button
        if st.button("Add to Dataset") and uploaded_file is not None and label:
            # Save the uploaded file
            file_path = save_audio_from_upload(uploaded_file)
            
            # Add to dataset
            st.session_state.data_manager.add_file(file_path, label)
            
            st.success(f"File added to dataset with label: {label}")
        
        # Display current dataset
        st.subheader("Current Dataset")
        
        if st.session_state.data_manager.files:
            # Show dataset summary
            df = st.session_state.data_manager.get_dataset_summary()
            
            # Display counts by label
            label_counts = df["label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]
            
            st.write("Files per label:")
            st.write(label_counts)
            
            # Display dataset table
            st.write("Dataset files:")
            st.dataframe(df)
        else:
            st.info(
                "No files added yet. Upload audio files and assign labels to build your dataset."
            )
    
    # Training tab
    with train_tab:
        st.subheader("Train a Classifier")
        
        # Model selection
        classifier_type = st.radio(
            "Classifier type:",
            ["K-Nearest Neighbors (KNN)", "Logistic Regression"]
        )
        
        clf_type = "knn" if "KNN" in classifier_type else "logistic"
        
        # Training button
        if st.button("Train Model"):
            if len(st.session_state.data_manager.files) < 5:
                st.error("Please add at least 5 audio files (ideally 10+) to the dataset.")
            elif len(np.unique(st.session_state.data_manager.labels)) < 2:
                st.error("Please add at least 2 different class labels.")
            else:
                with st.status("Training classifier..."):
                    # Train the model
                    model_data = st.session_state.data_manager.train_model(
                        classifier_type=clf_type
                    )
                
                # Display training results
                st.success(f"Model trained successfully! Accuracy: {model_data['accuracy']:.2f}")
                
                # Display confusion matrix
                cm = model_data["confusion_matrix"]
                classes = model_data["classes"]
                
                st.subheader("Confusion Matrix")
                cm_fig = dsp_plots.plot_confusion_matrix(cm, classes)
                st.plotly_chart(cm_fig)
                
                # Display model info
                st.write("Model Details:")
                st.json({
                    "classifier_type": clf_type,
                    "num_samples": len(st.session_state.data_manager.files),
                    "classes": list(classes),
                    "accuracy": float(model_data["accuracy"])
                })
    
    # Prediction tab
    with predict_tab:
        st.subheader("Predict New Audio")
        
        # Check if model exists
        if st.session_state.data_manager.model_data is None:
            st.info("Please train a model first before making predictions.")
        else:
            # File upload for prediction
            pred_file = st.file_uploader(
                "Upload a WAV file to classify:",
                type=["wav"],
                key="prediction_file_upload"
            )
            
            if pred_file is not None:
                # Save the uploaded file
                file_path = save_audio_from_upload(pred_file)
                
                # Make prediction button
                if st.button("Classify Audio"):
                    with st.status("Analyzing audio..."):
                        # Predict class
                        prediction = st.session_state.data_manager.predict(file_path)
                    
                    # Display prediction result
                    st.success(f"Prediction: {prediction}")
                    
                    # Display audio player
                    st.audio(file_path)
                    
                    # Load and display waveform
                    audio_data, sample_rate = load_audio_cached(file_path)
                    waveform_fig = dsp_plots.plot_waveform(
                        audio_data, 
                        sample_rate, 
                        title=f"Waveform (Predicted: {prediction})"
                    )
                    st.plotly_chart(waveform_fig, use_container_width=True)
                    
                    # Extract and show features
                    features = compute_features_cached(audio_data, sample_rate, file_path)
                    
                    # Show MFCC features
                    mfcc = features["mfcc"]["features"]
                    mfcc_times = features["mfcc"]["times"]
                    
                    mfcc_fig = dsp_plots.plot_mfcc(
                        mfcc, 
                        mfcc_times, 
                        title="MFCC Features Used for Classification"
                    )
                    st.plotly_chart(mfcc_fig, use_container_width=True)


def sidebar_file_browser() -> Optional[str]:
    """
    Create a file browser in the sidebar for audio files.
    
    Returns:
        Optional path to the selected file
    """
    st.sidebar.title("File Browser")
    
    # Choose directory to browse
    directory_options = {
        "Recordings": RECORDINGS_DIR,
        "Uploads": UPLOADS_DIR,
        "Synthesized": SYNTH_DIR,
        "All Files": None
    }
    
    selected_dir_name = st.sidebar.radio(
        "Browse audio files:",
        options=list(directory_options.keys())
    )
    
    selected_dir = directory_options[selected_dir_name]
    
    # Get list of audio files
    audio_files = []
    
    if selected_dir_name == "All Files":
        # Collect files from all directories
        for directory in ALL_AUDIO_DIRS:
            if directory.exists():
                for file in directory.glob("*.wav"):
                    audio_files.append(file)
    else:
        # Get files from the selected directory
        if selected_dir.exists():
            audio_files = list(selected_dir.glob("*.wav"))
    
    # Sort files by modified time (newest first)
    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Handle empty directory
    if not audio_files:
        st.sidebar.info(f"No audio files found in the {selected_dir_name.lower()} folder.")
        return None
    
    # Display file selection dropdown
    selected_file_path = None
    
    # Create a dictionary mapping filenames to their full paths
    # Also include the directory name in the display
    file_options = {}
    for file in audio_files:
        parent_dir = file.parent.name
        display_name = f"[{parent_dir}] {file.name}"
        file_options[display_name] = file
    
    selected_display = st.sidebar.selectbox(
        "Select an audio file:",
        options=list(file_options.keys()),
        format_func=lambda x: f"{x} ({format_file_size(os.path.getsize(file_options[x]))})"
    )
    
    if selected_display:
        selected_file = file_options[selected_display]
        selected_file_path = str(selected_file)
        
        # File operations section
        with st.sidebar.expander("File Operations"):
            # Display current filename
            st.write(f"**Current name:** {selected_file.name}")
            st.write(f"**Directory:** {selected_file.parent.name}")
            
            # Rename file option
            new_name = st.text_input("New filename (with .wav):", value=selected_file.name)
            
            # Move file option
            target_dir_options = [d for d in directory_options.keys() if d != "All Files"]
            target_dir = st.selectbox(
                "Move to directory:",
                options=target_dir_options,
                index=target_dir_options.index(selected_file.parent.name.capitalize())
            )
            
            col1, col2 = st.columns(2)
            
            # Rename button
            with col1:
                rename = st.button("Rename File")
                if rename and new_name != selected_file.name:
                    if not new_name.endswith(".wav"):
                        new_name += ".wav"
                    
                    # Check if new name already exists
                    if os.path.exists(selected_file.parent / new_name):
                        st.error(f"A file named '{new_name}' already exists.")
                    else:
                        # Rename the file
                        os.rename(selected_file, selected_file.parent / new_name)
                        st.success(f"File renamed to {new_name}")
                        st.rerun()  # Refresh the app to show the updated filename
            
            # Move button
            with col2:
                move = st.button("Move File")
                if move and target_dir.lower() != selected_file.parent.name:
                    target_path = directory_options[target_dir]
                    
                    # Check if file with same name exists in target directory
                    if os.path.exists(target_path / selected_file.name):
                        st.error(f"A file with the same name already exists in {target_dir}.")
                    else:
                        # Move the file
                        shutil.move(str(selected_file), str(target_path / selected_file.name))
                        st.success(f"File moved to {target_dir}")
                        st.rerun()  # Refresh the app to show the updated location
            
            # Play selected file
            st.audio(selected_file_path)
            
            # Load selected file button
            if st.button("Load Selected File"):
                return selected_file_path
    
    return None


def format_file_size(size_bytes):
    """Format file size in a human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def main():
    """Main application function."""
    # Ensure directories exist and organize files
    create_directories()
    organize_audio_files()
    
    st.title("AudioWave: Audio Signal Processing Explorer")
    
    # Introductory text
    st.markdown(
        """
        Explore audio signals through visualization and analysis. 
        Upload a sound file or record audio to get started!
        """
    )
    
    # Get audio input from sidebar
    sidebar_audio_input()
    
    # Add synthesizer to sidebar
    sidebar_synth()
    
    # Add file browser to sidebar and get selected file
    selected_file_path = sidebar_file_browser()
    
    # If a file was selected to load, update the session state
    if selected_file_path:
        try:
            # Load the audio file
            audio_data, sample_rate = load_audio_cached(selected_file_path)
            
            # Store in session state
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            st.session_state.source_path = selected_file_path
            
            # Display success message
            st.sidebar.success(f"File loaded successfully!")
            
            # Force a rerun to update the UI
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error loading audio file: {e}")
    
    # Use audio data from session state
    audio_data = st.session_state.audio_data
    sample_rate = st.session_state.sample_rate
    source_path = st.session_state.source_path
    
    # Clear audio data button
    if audio_data is not None:
        if st.sidebar.button("Clear Audio"):
            st.session_state.audio_data = None
            st.session_state.sample_rate = None
            st.session_state.source_path = None
            st.rerun()
    
    # Main content area
    if audio_data is not None and sample_rate is not None:
        # Compute features
        features = compute_features_cached(audio_data, sample_rate, source_path)
        
        # Create tabs with Dashboard as first tab
        dashboard, waveform, synth, spectrum, spectrogram, waterfall, samples, classification = st.tabs([
            "📊 Dashboard",
            "Waveform", 
            "Synth Sinusoid", 
            "Spectrum", 
            "STFT / Spectrogram",
            "Waterfall", 
            "Analysis (Kaggle sample)", 
            "Classification"
        ])
        
        # Fill tabs with content
        with dashboard:
            dashboard_tab(audio_data, sample_rate, features)
        
        with waveform:
            waveform_tab(audio_data, sample_rate)
        
        with synth:
            synth_tab()
        
        with spectrum:
            spectrum_tab(audio_data, sample_rate, features)
        
        with spectrogram:
            spectrogram_tab(audio_data, sample_rate, features)
        
        with waterfall:
            waterfall_tab(audio_data, sample_rate, features)
        
        with samples:
            sample_analysis_tab()
        
        with classification:
            classification_tab()
    
    else:
        # Display when no audio is loaded
        st.info(
            "Please upload an audio file or record audio using the sidebar controls to begin."
        )
        
        # Show placeholder content for tabs
        placeholder_tabs = st.tabs([
            "📊 Dashboard",
            "Waveform", 
            "Synth Sinusoid", 
            "Spectrum", 
            "STFT / Spectrogram",
            "Waterfall", 
            "Analysis (Kaggle sample)", 
            "Classification"
        ])
        
        # Synth tab works without input audio
        with placeholder_tabs[2]:
            synth_tab()


if __name__ == "__main__":
    main() 