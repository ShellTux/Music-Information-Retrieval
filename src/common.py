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
    tempo: np.ndarray

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
        features.mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features.spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Extract temporal features
        features.f0 = librosa.yin(y, fmin=20, fmax=sr/2)
        # Clean F0 values as in mrs.py
        features.f0[features.f0 == sr/2] = 0
        
        features.rms = librosa.feature.rms(y=y)
        features.zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        
        # Extract tempo
        features.tempo = librosa.feature.tempo(y=y, sr=sr)

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

    def _get_7_stats(self, data_array: np.ndarray) -> np.ndarray:
        """Helper function to calculate 7 statistics for a given 1D data array."""
        data_array = np.asarray(data_array).flatten() 
        if data_array.size == 0:
            # Return zeros if feature data is empty to maintain array shape
            # Consider more sophisticated handling if empty arrays are unexpected
            return np.zeros(7)
        
        # Replace NaNs and Infs that might come from librosa features or f0 processing
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.mean(data_array)
        std = np.std(data_array)
        # Check for constant array before skew/kurtosis to avoid warnings/NaNs
        if np.all(data_array == data_array[0] if data_array.size > 0 else True):
            skewness = 0.0
            kurt = -3.0 # Kurtosis of a constant is typically -3 (normal distribution = 0)
        else:
            skewness = stats.skew(data_array)
            kurt = stats.kurtosis(data_array)
        median = np.median(data_array)
        maximum = np.max(data_array)
        minimum = np.min(data_array)
        return np.array([mean, std, skewness, kurt, median, maximum, minimum])

    def calculate_statistics(self) -> np.ndarray:
        all_song_features_stats = []

        for i, song in enumerate(self.songs):
            f = song.features
            current_song_stats = []

            # MFCC (13 bands) -> 13 * 7 = 91 features
            for band_idx in range(f.mfcc.shape[0]):
                current_song_stats.extend(self._get_7_stats(f.mfcc[band_idx, :]))
            
            # Spectral Centroid -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_centroid))
            
            # Spectral Bandwidth -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_bandwidth))
            
            # Spectral Contrast (librosa default is 7 rows) -> 7 * 7 = 49 features
            for band_idx in range(f.spectral_contrast.shape[0]):
                current_song_stats.extend(self._get_7_stats(f.spectral_contrast[band_idx, :]))

            # Spectral Flatness -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_flatness))
            
            # Spectral Rolloff -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_rolloff))
            
            # F0 -> 7 features
            current_song_stats.extend(self._get_7_stats(f.f0))
            
            # RMS -> 7 features
            current_song_stats.extend(self._get_7_stats(f.rms))
            
            # Zero Crossing Rate -> 7 features
            current_song_stats.extend(self._get_7_stats(f.zero_crossing_rate))
            
            # Tempo -> 1 feature (just the value)
            # Ensure tempo is a scalar; librosa.feature.tempo returns an array (usually with one element)
            tempo_value = f.tempo[0] if isinstance(f.tempo, np.ndarray) and f.tempo.size > 0 else (f.tempo if np.isscalar(f.tempo) else 0)
            current_song_stats.append(tempo_value)
            
            all_song_features_stats.append(current_song_stats)
            
        return np.array(all_song_features_stats)

    def normalize_features(self, statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_vals = np.min(statistics, axis=0)
        max_vals = np.max(statistics, axis=0)
        
        # Avoid division by zero: if max == min, feature is constant, normalized value should be 0.
        # (X - min) / (max - min). If max - min is 0, then X == min, so X - min is 0.
        # We set denominator to 1 in this case, so 0 / 1 = 0.
        range_vals = max_vals - min_vals
        # Create a safe version of range_vals for division
        # Where range_vals is 0, use 1 to avoid division by zero (numerator will also be 0)
        # Where range_vals is not 0, use actual range_vals
        safe_range_vals = np.where(range_vals == 0, 1, range_vals)
        
        normalized_statistics = (statistics - min_vals) / safe_range_vals
        
        # Ensure any NaNs that might arise (e.g., if a column was all NaNs initially, though _get_7_stats tries to prevent this)
        # are converted to numbers (e.g., 0).
        normalized_statistics = np.nan_to_num(normalized_statistics, nan=0.0)

        return normalized_statistics, min_vals, max_vals

    def save_features_to_file(self, filename: str | Path):
        unnormalized_statistics = self.calculate_statistics()
        
        if unnormalized_statistics.size == 0:
            print("No statistics were calculated. Cannot save to file.", file=stderr)
            return

        normalized_stats, min_vals, max_vals = self.normalize_features(unnormalized_statistics)

        # Create the output array with min_vals, max_vals, and then normalized feature statistics
        output_array = np.vstack([min_vals, max_vals, normalized_stats])

        # Save to file, without a header, matching mrs.py's typical output for feature files
        np.savetxt(filename, output_array, delimiter=',', fmt='%.6f', comments='')

        print(f"Feature statistics (min, max, normalized) saved to {filename}")

    @staticmethod
    def load_features_from_file(filename: str | Path) -> np.ndarray:
        data = np.loadtxt(filename, delimiter=',')

        # Check that we have the expected number of rows
        if data.shape[0] < 3:
            raise ValueError("File must contain at least min, max, and one row of statistics.")

        return data # Return all data: min_vals, max_vals, and normalized_stats
        # The old code returned data[2:,:], which would only be normalized_stats

    def print(self):
        if len(self.songs) == 0:
            print("No songs found.")
        else:
            print("List of MP3 Songs:")
            for song in self.songs:
                print(song.filename)
