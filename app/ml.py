"""
Machine learning utilities for audio feature extraction and classification.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def extract_features(audio_file: Union[str, Path], 
                    sr: Optional[int] = None,
                    n_mfcc: int = 20) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    
    Args:
        audio_file: Path to the audio file
        sr: Sample rate for loading audio. If None, uses the file's native sample rate.
        n_mfcc: Number of MFCC coefficients to extract
        
    Returns:
        Mean MFCC features across time frames
    """
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr, mono=True)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Compute mean of MFCCs across time
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean


def extract_dataset_features(audio_files: List[str], 
                           labels: List[str],
                           sr: Optional[int] = None,
                           n_mfcc: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a list of audio files with corresponding labels.
    
    Args:
        audio_files: List of audio file paths
        labels: List of corresponding labels
        sr: Sample rate for loading audio
        n_mfcc: Number of MFCC coefficients to extract
        
    Returns:
        Tuple of (features_array, labels_array)
    """
    features = []
    
    for audio_file in audio_files:
        # Extract features
        mfccs_mean = extract_features(audio_file, sr=sr, n_mfcc=n_mfcc)
        features.append(mfccs_mean)
    
    # Convert to numpy arrays
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    return features_array, labels_array


def train_classifier(features: np.ndarray, 
                    labels: np.ndarray,
                    classifier_type: str = "knn", 
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict[str, Any]:
    """
    Train a classifier on audio features.
    
    Args:
        features: Array of feature vectors
        labels: Array of corresponding labels
        classifier_type: Type of classifier to train ("knn" or "logistic")
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing the trained model, scaler, and evaluation metrics
    """
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize classifier
    if classifier_type.lower() == "knn":
        classifier = KNeighborsClassifier(n_neighbors=3)
    else:  # logistic regression
        classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    
    # Train the classifier
    classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Return results
    return {
        "model": classifier,
        "scaler": scaler,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classes": np.unique(labels),
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred
    }


def predict_audio(model: Any, 
                 scaler: Any,
                 audio_file: Union[str, Path],
                 sr: Optional[int] = None,
                 n_mfcc: int = 20) -> str:
    """
    Predict the class of an audio file using a trained model.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        audio_file: Path to the audio file
        sr: Sample rate for loading audio
        n_mfcc: Number of MFCC coefficients to extract
        
    Returns:
        Predicted class label
    """
    # Extract features
    features = extract_features(audio_file, sr=sr, n_mfcc=n_mfcc)
    
    # Reshape for scaling (as we have a single sample)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return prediction


class AudioDataManager:
    """
    Class to manage audio files, their labels, and feature extraction.
    """
    
    def __init__(self, audio_dir: str = "audio_inputs"):
        """
        Initialize the audio data manager.
        
        Args:
            audio_dir: Directory containing audio files
        """
        self.audio_dir = audio_dir
        self.files = []
        self.labels = []
        self.model_data = None
    
    def add_file(self, file_path: str, label: str) -> None:
        """
        Add a file to the dataset.
        
        Args:
            file_path: Path to the audio file
            label: Class label for the file
        """
        self.files.append(file_path)
        self.labels.append(label)
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """
        Get a summary of the dataset.
        
        Returns:
            DataFrame with file paths and labels
        """
        return pd.DataFrame({
            "file": self.files,
            "label": self.labels
        })
    
    def extract_features(self, n_mfcc: int = 20, sr: Optional[int] = None) -> np.ndarray:
        """
        Extract features from all files in the dataset.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            sr: Sample rate for loading audio
            
        Returns:
            Array of feature vectors
        """
        features, labels = extract_dataset_features(
            self.files, self.labels, sr=sr, n_mfcc=n_mfcc
        )
        return features
    
    def train_model(self, classifier_type: str = "knn", test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a model on the dataset.
        
        Args:
            classifier_type: Type of classifier to train ("knn" or "logistic")
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        # Extract features
        features = self.extract_features()
        
        # Train classifier
        self.model_data = train_classifier(
            features, np.array(self.labels), 
            classifier_type=classifier_type,
            test_size=test_size
        )
        
        return self.model_data
    
    def predict(self, audio_file: str) -> str:
        """
        Predict the class of an audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Predicted class label
        """
        if self.model_data is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        return predict_audio(
            self.model_data["model"],
            self.model_data["scaler"],
            audio_file
        ) 