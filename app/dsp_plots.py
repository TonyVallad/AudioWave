"""
Visualization functions for audio signals and features using Plotly.
"""
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa


def plot_waveform(audio_data: np.ndarray, 
                 sample_rate: int, 
                 title: str = "Audio Waveform") -> go.Figure:
    """
    Plot the time-domain representation of an audio signal.
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create time array
    t = np.arange(len(audio_data)) / sample_rate
    
    # Create figure
    fig = go.Figure()
    
    # Add waveform trace
    fig.add_trace(
        go.Scatter(
            x=t,
            y=audio_data,
            mode='lines',
            name='Amplitude',
            line=dict(color='#1f77b4', width=1)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Add grid and zoom tools
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_spectrum(frequencies: np.ndarray, 
                 magnitudes: np.ndarray, 
                 power_db: Optional[np.ndarray] = None,
                 title: str = "Audio Spectrum",
                 log_freq: bool = False) -> go.Figure:
    """
    Plot the frequency spectrum of an audio signal.
    
    Args:
        frequencies: Frequency bins from FFT
        magnitudes: Linear magnitude values
        power_db: Optional power in dB
        title: Plot title
        log_freq: Whether to use log scale for frequency axis
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Get values to plot (dB or linear)
    y_values = power_db if power_db is not None else magnitudes
    y_axis_title = "Power (dB)" if power_db is not None else "Magnitude"
    
    # Add spectrum trace
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=y_values,
            mode='lines',
            name=y_axis_title,
            line=dict(color='#ff7f0e', width=1.5)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_axis_title,
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Set log scale for frequency if requested
    if log_freq:
        fig.update_xaxes(type="log")
    
    # Add grid and zoom tools
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_spectrogram(spectrogram: np.ndarray, 
                    frequencies: np.ndarray, 
                    times: np.ndarray,
                    title: str = "Spectrogram") -> go.Figure:
    """
    Plot a spectrogram (STFT).
    
    Args:
        spectrogram: 2D array of spectrogram values (frequency bins x time frames)
        frequencies: Frequency bins
        times: Time frames
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add spectrogram as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=spectrogram,
            x=times,
            y=frequencies,
            colorscale='Viridis',
            zmin=np.min(spectrogram),
            zmax=np.max(spectrogram),
            colorbar=dict(title="dB")
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


def plot_mel_spectrogram(mel_spectrogram: np.ndarray, 
                        mel_frequencies: np.ndarray, 
                        times: np.ndarray,
                        title: str = "Mel Spectrogram") -> go.Figure:
    """
    Plot a Mel spectrogram.
    
    Args:
        mel_spectrogram: 2D array of Mel spectrogram values
        mel_frequencies: Mel frequency bins
        times: Time frames
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add Mel spectrogram as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=mel_spectrogram,
            x=times,
            y=mel_frequencies,
            colorscale='Viridis',
            zmin=np.min(mel_spectrogram),
            zmax=np.max(mel_spectrogram),
            colorbar=dict(title="dB")
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Mel Frequency (Hz)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Use log scale for frequency axis
    fig.update_yaxes(type="log")
    
    return fig


def plot_mfcc(mfcc_features: np.ndarray, 
             times: np.ndarray,
             title: str = "MFCC") -> go.Figure:
    """
    Plot Mel-frequency cepstral coefficients.
    
    Args:
        mfcc_features: 2D array of MFCC values
        times: Time frames
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add MFCCs as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=mfcc_features,
            x=times,
            y=np.arange(mfcc_features.shape[0]),
            colorscale='Viridis',
            colorbar=dict(title="Coefficient\nValue")
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="MFCC Coefficient",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


def plot_waterfall(spectrogram: np.ndarray, 
                  frequencies: np.ndarray, 
                  times: np.ndarray,
                  title: str = "Waterfall Plot") -> go.Figure:
    """
    Create a 3D waterfall plot of a spectrogram with time flowing vertically.
    
    Args:
        spectrogram: 2D array of spectrogram values (frequency bins x time frames)
        frequencies: Frequency bins
        times: Time frames
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Normalize color values
    # Convert to dB scale relative to maximum
    spec_norm = spectrogram - np.max(spectrogram)
    
    # Create surface plot with time flowing vertically (y-axis)
    fig.add_trace(
        go.Surface(
            z=spec_norm,
            x=frequencies,
            y=times,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Power (dB)"),
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.2),
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Time (s)",
            zaxis_title="Power (dB)",
            # Default camera position for good view
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
            ),
            aspectratio=dict(x=1, y=1, z=0.4),
            # Make grid visible
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        ),
        template="plotly_white",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


def plot_waterfall_2d(spectrogram: np.ndarray, 
                     frequencies: np.ndarray, 
                     times: np.ndarray,
                     title: str = "2D Waterfall Plot") -> go.Figure:
    """
    Create a 2D waterfall plot with frequency on x-axis, time flowing vertically from bottom to top on y-axis,
    and color representing power.
    
    Args:
        spectrogram: 2D array of spectrogram values (frequency bins x time frames)
        frequencies: Frequency bins
        times: Time frames
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add heatmap with time flowing from bottom to top
    fig.add_trace(
        go.Heatmap(
            z=spectrogram,
            x=frequencies,
            y=times,
            colorscale='Viridis',
            zmin=np.min(spectrogram),
            zmax=np.max(spectrogram),
            colorbar=dict(title="Power (dB)"),
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Time (s)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        # Reverse y-axis so time flows from bottom to top
        yaxis=dict(autorange="reversed")
    )
    
    # Add grid and zoom tools
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_waveform_and_features(audio_data: np.ndarray,
                              sample_rate: int,
                              stft: Optional[np.ndarray] = None,
                              mel_spec: Optional[np.ndarray] = None,
                              mfcc: Optional[np.ndarray] = None,
                              title: str = "Audio Analysis") -> go.Figure:
    """
    Create a multi-panel plot with waveform and audio features.
    
    Args:
        audio_data: Audio time-domain signal
        sample_rate: Sample rate in Hz
        stft: STFT spectrogram (optional)
        mel_spec: Mel spectrogram (optional)
        mfcc: MFCC features (optional)
        title: Main plot title
    
    Returns:
        Plotly figure object with subplots
    """
    # Count how many features to plot
    num_features = 1  # Always include waveform
    if stft is not None:
        num_features += 1
    if mel_spec is not None:
        num_features += 1
    if mfcc is not None:
        num_features += 1
    
    # Create subplot figure
    fig = make_subplots(
        rows=num_features,
        cols=1,
        subplot_titles=["Waveform"],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add waveform
    t = np.arange(len(audio_data)) / sample_rate
    fig.add_trace(
        go.Scatter(
            x=t,
            y=audio_data,
            mode='lines',
            name='Amplitude',
            line=dict(color='#1f77b4', width=1)
        ),
        row=1, col=1
    )
    
    # Row counter for additional features
    current_row = 2
    
    # Add STFT if provided
    if stft is not None:
        # Add to subplot titles
        fig.layout.annotations[current_row-1].text = "STFT Spectrogram"
        
        # Compute time and frequency axes
        times = np.linspace(0, len(audio_data) / sample_rate, stft.shape[1])
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2 * (stft.shape[0] - 1))
        
        # Add spectrogram
        fig.add_trace(
            go.Heatmap(
                z=stft,
                x=times,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB", y=(current_row - 0.5) / num_features, len=1/num_features)
            ),
            row=current_row, col=1
        )
        
        # Move to next row
        current_row += 1
    
    # Add Mel spectrogram if provided
    if mel_spec is not None:
        # Add to subplot titles
        fig.layout.annotations[current_row-1].text = "Mel Spectrogram"
        
        # Compute time and mel frequency axes
        times = np.linspace(0, len(audio_data) / sample_rate, mel_spec.shape[1])
        mel_freqs = librosa.mel_frequencies(n_mels=mel_spec.shape[0])
        
        # Add mel spectrogram
        fig.add_trace(
            go.Heatmap(
                z=mel_spec,
                x=times,
                y=mel_freqs,
                colorscale='Viridis',
                colorbar=dict(title="dB", y=(current_row - 0.5) / num_features, len=1/num_features)
            ),
            row=current_row, col=1
        )
        
        # Move to next row
        current_row += 1
    
    # Add MFCCs if provided
    if mfcc is not None:
        # Add to subplot titles
        fig.layout.annotations[current_row-1].text = "MFCC"
        
        # Compute time axis
        times = np.linspace(0, len(audio_data) / sample_rate, mfcc.shape[1])
        
        # Add MFCCs
        fig.add_trace(
            go.Heatmap(
                z=mfcc,
                x=times,
                y=np.arange(mfcc.shape[0]),
                colorscale='Viridis',
                colorbar=dict(title="Value", y=(current_row - 0.5) / num_features, len=1/num_features)
            ),
            row=current_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300 * num_features,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    # Label axes
    fig.update_xaxes(title_text="Time (s)", row=num_features, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    
    if stft is not None:
        row = 2
        fig.update_yaxes(title_text="Frequency (Hz)", row=row, col=1)
        if mel_spec is not None:
            row += 1
            fig.update_yaxes(title_text="Mel Frequency (Hz)", row=row, col=1)
        if mfcc is not None:
            row += 1
            fig.update_yaxes(title_text="MFCC Coefficient", row=row, col=1)
    elif mel_spec is not None:
        row = 2
        fig.update_yaxes(title_text="Mel Frequency (Hz)", row=row, col=1)
        if mfcc is not None:
            row += 1
            fig.update_yaxes(title_text="MFCC Coefficient", row=row, col=1)
    elif mfcc is not None:
        fig.update_yaxes(title_text="MFCC Coefficient", row=2, col=1)
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         title: str = "Confusion Matrix") -> go.Figure:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        class_names: List of class names
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm.astype(int),
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverinfo='text'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label", autorange="reversed"),
        template="plotly_white",
        height=450,
        width=450,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig 