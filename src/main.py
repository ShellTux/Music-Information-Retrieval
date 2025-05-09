from common import BD
from scipy.fft import rfft
from scipy.stats import pearsonr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.spatial import distance

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

    # --- ALÍNEA 3: Implementação de Métricas de Similaridade ---
    print("\n--- Iniciando Alínea 3: Métricas de Similaridade ---")

    # Carregar os dados de features_statistics.csv (min, max, e estatísticas não normalizadas)
    try:
        data_from_file = np.loadtxt(features_path, delimiter=',', skiprows=1) # skiprows=1 para pular o header
        min_vals_dataset = data_from_file[0, :]
        max_vals_dataset = data_from_file[1, :]
        unnormalized_stats_all_songs = data_from_file[2:, :]
    except FileNotFoundError:
        print(f"ERRO: O ficheiro {features_path} não foi encontrado. Execute a primeira parte do script para gerá-lo.")
        return
    except IndexError:
        print(f"ERRO: O ficheiro {features_path} não parece conter os dados esperados (min, max, stats).")
        return

    # Normalizar todas as estatísticas das músicas do dataset
    range_vals_ds = max_vals_dataset - min_vals_dataset
    # Evitar divisão por zero: onde o range é 0, o valor normalizado será 0.
    # (X - min) / range. Se range é 0, min == max, então X-min é 0. 0/0 -> NaN.
    # Se range_vals_ds[i] == 0, significa que min_val == max_val para essa feature.
    # Nesse caso, (unnormalized_value - min_val) / (non_zero_range_placeholder) will be 0 / placeholder.
    # Se definirmos o placeholder como 1, o resultado é 0, o que é correto para uma feature sem variação.
    range_vals_ds_safe = np.where(range_vals_ds == 0, 1, range_vals_ds)
    normalized_stats_all_songs = (unnormalized_stats_all_songs - min_vals_dataset) / range_vals_ds_safe
    normalized_stats_all_songs = np.nan_to_num(normalized_stats_all_songs) # Segurança extra para NaNs

    # Obter a lista ordenada de ficheiros de música (para mapear índices para nomes)
    # Esta lógica deve corresponder à da classe BD em common.py
    song_files_ordered = []
    data_dir = './MER_audio_taffc_dataset' # Diretório das músicas
    for root_dir, _, files_in_dir in os.walk(data_dir):
        for file_item in files_in_dir:
            if file_item.endswith('.mp3'):
                song_files_ordered.append(Path(os.path.join(root_dir, file_item)))
    song_files_ordered.sort(key=lambda p: (p.parent.name, p.name))

    if not song_files_ordered:
        print(f"ERRO: Nenhum ficheiro MP3 encontrado em {data_dir}.")
        return

    if len(song_files_ordered) != normalized_stats_all_songs.shape[0]:
        print(f"AVISO: Número de ficheiros MP3 ({len(song_files_ordered)}) não corresponde ao número de registos de estatísticas ({normalized_stats_all_songs.shape[0]}). As recomendações podem estar desalinhadas.")
        # Poderia truncar song_files_ordered para o tamanho de normalized_stats_all_songs se soubermos que é o correto.
        # Ou parar se for um erro crítico. Por agora, apenas um aviso.
        if not normalized_stats_all_songs.any(): # No stats loaded
             print("Abortando devido à falta de estatísticas.")
             return

    # 3.2.1. Para a query: Selecionar uma música query (ex: a primeira da lista)
    query_song_index = 28
    if query_song_index >= len(song_files_ordered) or query_song_index >= normalized_stats_all_songs.shape[0]:
        print(f"ERRO: Índice da música query ({query_song_index}) fora dos limites. Certifique-se que tem pelo menos {query_song_index + 1} músicas no dataset.")
        return

    query_song_normalized_stats = normalized_stats_all_songs[query_song_index]
    query_song_filename = song_files_ordered[query_song_index]
    print(f"Música Query: {query_song_filename.name}")

    # 3.1. Métricas de similaridade já estão disponíveis em scipy.spatial.distance

    # 3.2.2. Criar e gravar em ficheiro 3 matrizes de similaridade (vetores de distância)
    num_songs_in_db = normalized_stats_all_songs.shape[0]

    distances_euclidean = np.zeros(num_songs_in_db)
    distances_manhattan = np.zeros(num_songs_in_db)
    distances_cosine = np.zeros(num_songs_in_db)

    for i in range(num_songs_in_db):
        db_song_normalized_stats = normalized_stats_all_songs[i]
        distances_euclidean[i] = distance.euclidean(query_song_normalized_stats, db_song_normalized_stats)
        distances_manhattan[i] = distance.cityblock(query_song_normalized_stats, db_song_normalized_stats) # cityblock é Manhattan
        distances_cosine[i] = distance.cosine(query_song_normalized_stats, db_song_normalized_stats) # 1 - similaridade cosseno

    # Guardar as matrizes de similaridade (vetores de distância)
    np.savetxt('similarity_euclidean.csv', distances_euclidean.reshape(-1, 1), delimiter=',', header='EuclideanDistance', comments='', fmt='%.6f')
    np.savetxt('similarity_manhattan.csv', distances_manhattan.reshape(-1, 1), delimiter=',', header='ManhattanDistance', comments='', fmt='%.6f')
    np.savetxt('similarity_cosine.csv', distances_cosine.reshape(-1, 1), delimiter=',', header='CosineDistance', comments='', fmt='%.6f')
    print("Matrizes de similaridade (vetores de distância) guardadas em .csv")

    # 3.3. Para a query, criar os 3 rankings de similaridade (top 10)
    recommendation_count = 10

    # Ranking Euclidiano
    sorted_indices_euclidean = np.argsort(distances_euclidean)
    # Excluir a própria música query (índice 0 do sortido) e pegar as próximas 10
    recommended_indices_euclidean = sorted_indices_euclidean[1 : recommendation_count + 1]

    # Ranking Manhattan
    sorted_indices_manhattan = np.argsort(distances_manhattan)
    recommended_indices_manhattan = sorted_indices_manhattan[1 : recommendation_count + 1]

    # Ranking Cosseno
    sorted_indices_cosine = np.argsort(distances_cosine)
    recommended_indices_cosine = sorted_indices_cosine[1 : recommendation_count + 1]

    print(f"\nTop {recommendation_count} Recomendações para {query_song_filename.name} (excluindo a própria música):")

    print("\nRecomendações - Distância Euclidiana:")
    for i, idx in enumerate(recommended_indices_euclidean):
        if idx < len(song_files_ordered): # Check bounds
            print(f"{i+1}. {song_files_ordered[idx].name} (Distância: {distances_euclidean[idx]:.4f})")
        else:
            print(f"{i+1}. Índice {idx} fora dos limites para nomes de ficheiro.")


    print("\nRecomendações - Distância Manhattan:")
    for i, idx in enumerate(recommended_indices_manhattan):
        if idx < len(song_files_ordered): # Check bounds
            print(f"{i+1}. {song_files_ordered[idx].name} (Distância: {distances_manhattan[idx]:.4f})")
        else:
            print(f"{i+1}. Índice {idx} fora dos limites para nomes de ficheiro.")

    print("\nRecomendações - Distância Cosseno:")
    for i, idx in enumerate(recommended_indices_cosine):
        if idx < len(song_files_ordered): # Check bounds
            print(f"{i+1}. {song_files_ordered[idx].name} (Distância: {distances_cosine[idx]:.4f})")
        else:
            print(f"{i+1}. Índice {idx} fora dos limites para nomes de ficheiro.")

    print("\n--- Alínea 3 Concluída ---")

if __name__ == '__main__':
    main()
