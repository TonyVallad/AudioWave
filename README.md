# AudioWave: Interactive Audio Processing Explorer

AudioWave is an educational Streamlit application for exploring audio signal processing concepts. It allows users to upload or record audio, visualize it in the time and frequency domains, synthesize simple sounds, and train a basic audio classifier.

## Features

- **Audio Input**: Upload WAV files or record directly from your microphone
- **Waveform Visualization**: View time-domain representation of audio signals
- **Sinusoid Synthesis**: Generate sine waves at different frequencies
- **Spectrum Analysis**: Compute and display FFT with interactive controls
- **Spectrogram Views**: Explore STFT and Mel spectrograms of audio signals
- **MFCC Extraction**: Visualize Mel-frequency cepstral coefficients
- **Audio Classification**: Build datasets and train KNN or logistic regression models
- **Interactive UI**: User-friendly interface with informative visualizations

## Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd AudioWave
   ```

2. Create and activate a virtual environment:

   **Windows:**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: On some systems, you might need additional system dependencies for PyAudio/PortAudio.

   **Windows:** PyAudio should install directly through pip.

   **macOS:**
   ```
   brew install portaudio
   pip install pyaudio
   ```

   **Linux (Ubuntu/Debian):**
   ```
   sudo apt-get install portaudio19-dev python-pyaudio
   pip install pyaudio
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app/main.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the sidebar to upload a WAV file or record audio from your microphone

4. Explore the different tabs to analyze your audio:
   - **Waveform**: View basic audio information and time-domain plot
   - **Synth Sinusoid**: Generate sine waves at different frequencies
   - **Spectrum**: Analyze frequency content with FFT
   - **STFT/Spectrogram**: View time-frequency representations
   - **Analysis**: Examine sample audio files with various features
   - **Classification**: Build a simple audio classifier

## Dataset Integration

For the classification feature, you can download and integrate standard audio datasets:

1. **UrbanSound8K**: Urban sound classification dataset with 10 classes
   - Download: [https://urbansounddataset.weebly.com/](https://urbansounddataset.weebly.com/)

2. **ESC-50**: Environmental sound classification dataset with 50 classes
   - Download: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

After downloading, you can use individual audio files from these datasets for classification experiments within the app.

## Project Structure

```
AudioWave/
├── app/
│   ├── main.py            # Streamlit UI
│   ├── audio_utils.py     # Audio loading, recording, DSP helpers
│   ├── dsp_plots.py       # Visualization functions using Plotly
│   ├── ml.py              # MFCC extraction & classifier
├── audio_inputs/          # Directory for uploaded/recorded audio
├── notebooks/             # Optional Jupyter notebooks
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Educational Concepts

This application demonstrates several key concepts in audio signal processing:

- Time-domain and frequency-domain representations
- Fourier analysis (FFT and STFT)
- Spectrograms and time-frequency analysis
- Mel frequency scale and human auditory perception
- Feature extraction with MFCCs
- Audio classification with machine learning

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Data sources: UrbanSound8K dataset and ESC-50 dataset
- Built with Streamlit, Librosa, NumPy, Plotly, and scikit-learn 