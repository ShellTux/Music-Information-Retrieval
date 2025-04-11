from common import BD
from scipy.fft import rfft
from scipy.stats import pearsonr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

features_path = 'features_statistics.csv'

def compute_spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
    # Realiza a Transformada Rápida de Fourier
    magnitude_spectrum: np.ndarray = np.abs(rfft(y))

    # Calcular a frequência correspondente a cada bin
    bin_count = len(magnitude_spectrum)
    frequencies = np.linspace(0, sr / 2, bin_count)

    # Calcular o centróide espectral
    spectral_centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0

    return spectral_centroid

def main():
    if not os.path.exists(features_path):
        SONGS = BD('./MER_audio_taffc_dataset')
        statistics = SONGS.calculate_statistics()
        normalized_statistics = SONGS.normalize_features(statistics)
        print("Statistics Array:")
        print(statistics.round(2))
        print("Normalized Statistics Array:")
        print(normalized_statistics.round(2))
        SONGS.save_features_to_file(features_path)

    statistics = BD.load_features_from_file(features_path)

    # Inicialização
    sr = 22050  # taxa de amostragem
    duration = 5.0  # duração do sinal em segundos
    frame_length = 2048
    hop_length = 512

    # Listas para armazenar os resultados
    rmse_values = []
    pearson_corr_values = []

    # Gerar múltiplos sinais e calcular as métricas
    for _ in range(900):
        # Criar um sinal aleatório de sinusóides com frequências variando
        frequency1 = np.random.uniform(200, 800)  # Frequência 1 entre 200 e 800 Hz
        frequency2 = np.random.uniform(200, 800)  # Frequência 2 entre 200 e 800 Hz
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * frequency1 * t) + 0.5 * np.sin(2 * np.pi * frequency2 * t)

        # Calcular o centróide espectral com a implementação manual
        sc_manual = []
        for start in range(0, len(y), hop_length):
            end = min(start + frame_length, len(y))
            frame = y[start:end]
            if len(frame) == frame_length:
                sc_manual.append(compute_spectral_centroid(frame, sr))

        # Calcular o centróide espectral usando librosa
        sc_librosa = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
        sc_librosa = sc_librosa.flatten()

        # Alinhar resultados (2 janelas de atraso)
        sc_librosa = sc_librosa[2:]

        # HACK: Idk
        sc_librosa = sc_librosa[2:]

        # Calcular RMSE
        rmse = np.sqrt(np.mean((sc_manual - sc_librosa) ** 2))

        # Calcular o coeficiente de correlação de Pearson
        pearson_corr, _ = pearsonr(sc_manual, sc_librosa)

        # Armazenar os resultados
        rmse_values.append(rmse)
        pearson_corr_values.append(pearson_corr)

    # Criar um DataFrame com os resultados
    results_array = np.column_stack((rmse_values, pearson_corr_values))
    np.savetxt(
        'resultados_metricas.csv',
        results_array,
        delimiter=',',
        header='RMSE,Pearson_Correlation',
        comments='',
        fmt=['%.6f', '%.6f']
    )

    print("Resultados salvos em 'resultados_metricas.csv'")

if __name__ == '__main__':
    main()
