from pathlib import Path
import random
from scipy import stats
from sys import stderr
import librosa
import numpy as np
import os

class Features:
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_contrast: np.ndarray
    spectral_flatness: np.ndarray
    spectral_rolloff: np.ndarray
    f0: np.ndarray
    rms: np.ndarray
    zero_crossing_rate: np.ndarray
    time: np.ndarray

class Song:
    filename: Path
    features: Features = Features()

    def __init__(self, filepath: str | Path):
        self.filename = Path(filepath)
        assert self.filename.exists(), f"File not found: {self.filename}"

        self.extract_features()

    def extract_features(self) -> Features:
        y, sr = librosa.load(self.filename, sr=22050, mono=True)
        features = Features()

        # Extract spectral features
        # TODO: sr / 2
        features.mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features.spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Extract temporal features
        # NOTE: Takes too much time
        # features.f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        features.rms = librosa.feature.rms(y=y)
        features.zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        # Get time variable
        features.time = np.arange(len(y)) / sr

        self.features = features

        return features

class BD:
    songs: list[Song] = []

    def __init__(self, directory: str | Path):
        self.songs = []

        songs_path: list[Path] = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp3'):
                    songs_path.append(Path(os.path.join(root, file)))

        # songs_path = random.sample(songs_path, 30)
        songs_path.sort(key=lambda path: (path.parent.name, path.name))
        # songs_path = songs_path[:30]

        for i, path in enumerate(songs_path):
            self.songs.append(Song(path))
            index = i + 1
            ratio = index / len(songs_path) * 100
            print(f'\r{index:3} / {len(songs_path)} ({ratio:.2f}%) done. {path}', end='')

        print()

    def calculate_statistics(self):
        n_songs = len(self.songs)

        # Define the number of features to store statistics for:
        n_features = 7  # mean, std, skewness, kurtosis, median, max, min

        # Create a 2D numpy array
        stats_array = np.zeros((n_songs, n_features))

        for i, song in enumerate(self.songs):
            f = song.features  # Features for the song
            feature_data = []

            # Collect feature data for statistics
            feature_data.extend(np.mean(f.mfcc, axis=1))  # Mean of MFCC for each coefficient
            feature_data.append(np.mean(f.spectral_centroid))
            feature_data.append(np.mean(f.spectral_bandwidth))
            feature_data.append(np.mean(f.spectral_contrast))
            feature_data.append(np.mean(f.spectral_flatness))
            feature_data.append(np.mean(f.spectral_rolloff))
            feature_data.append(np.mean(f.rms))
            feature_data.append(np.mean(f.zero_crossing_rate))

            # Calculate statistics
            stats_array[i, 0] = np.mean(feature_data)
            stats_array[i, 1] = np.std(feature_data)
            stats_array[i, 2] = stats.skew(feature_data)
            stats_array[i, 3] = stats.kurtosis(feature_data)
            stats_array[i, 4] = np.median(feature_data)
            stats_array[i, 5] = np.max(feature_data)
            stats_array[i, 6] = np.min(feature_data)

        return stats_array

    def normalize_features(self, statistics: np.ndarray) -> np.ndarray:
        # Normalizing by calculating min and max for the given statistics array
        min_vals = statistics.min(axis=0)
        max_vals = statistics.max(axis=0)

        # Normalize each statistic
        normalized = (statistics - min_vals) / (max_vals - min_vals)

        return normalized

    def save_features_to_file(self, filename: str | Path):
        statistics = self.calculate_statistics()

        # Prepare to save minimum and maximum values
        min_vals = statistics.min(axis=0)
        max_vals = statistics.max(axis=0)

        # Create the output array with min, max, and feature statistics
        output_array = np.vstack([min_vals, max_vals, statistics])

        # Save to file
        # TODO: Remove header
        np.savetxt(filename, output_array, delimiter=',', header='Min,Max,Mean,Std,Skew,Kurtosis,Median,Max,Min', comments='')

        print(f"Feature statistics saved to {filename}")

    @staticmethod
    def load_features_from_file(filename: str | Path) -> np.ndarray:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)

        # Check that we have the expected number of rows
        if data.shape[0] < 3:
            raise ValueError("File must contain at least min, max, and one row of statistics.")

        return data[2:, :]


    def print(self):
        if len(self.songs) == 0:
            print("No songs found.")
        else:
            print("List of MP3 Songs:")
            for song in self.songs:
                print(song.filename)
