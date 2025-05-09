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
import pandas as pd

features_path = 'features_statistics.csv'

def compute_spectral_centroid(y: np.ndarray, sr: int, frame_length_for_fft: int) -> float:
    # Apply Hanning window
    if len(y) == 0:
        return 0.0
    window = np.hanning(len(y))
    y_windowed = y * window
    
    magnitude_spectrum: np.ndarray = np.abs(rfft(y_windowed))
    
    # Frequencies should correspond to the n_fft used, which is the length of the windowed frame
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length_for_fft)

    # Ensure frequencies array matches magnitude_spectrum length (rfft gives N/2+1 bins)
    # For rfft, the number of frequency bins is n_fft/2 + 1
    expected_bins = frame_length_for_fft // 2 + 1
    if len(magnitude_spectrum) != expected_bins:
        # This case should ideally not be hit if frame length for rfft input and n_fft for frequencies match
        # print(f"Warning: Mismatch in spectral centroid. Mag bins: {len(magnitude_spectrum)}, Freq bins from n_fft: {len(frequencies)}, expected from frame: {expected_bins}")
        # Fallback or adjustment might be needed if this occurs frequently.
        # For now, trim the longer array to match the shorter one, preferring magnitude_spectrum's length.
        frequencies = frequencies[:len(magnitude_spectrum)]

    spectral_centroid_val = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0.0
    return float(spectral_centroid_val)

def main():
    # --- ALÍNEA 2.1: Feature Extraction and Saving ---
    songs_bd_instance = None # To store the BD instance for later use
    if not os.path.exists(features_path):
        print(f"'{features_path}' not found. Generating features...")
        songs_bd_instance = BD('./MER_audio_taffc_dataset') 
        if not songs_bd_instance.songs:
            print("No songs were successfully processed by BD. Halting.")
            return
        songs_bd_instance.save_features_to_file(features_path)
        print(f"Features saved to '{features_path}'")
    else:
        print(f"Using existing features from '{features_path}'")
        # If features exist, we still need the ordered list of song names for consistency
        # Create a BD instance just to get this list, it won't reprocess songs if only names are needed.
        # However, to ensure `get_ordered_song_filenames` is populated correctly based on how `features_path`
        # *would have been* created, we should initialize BD fully.
        # This assumes that if features_path exists, it was created with the same ordering logic (CSV-based).
        print("Initializing BD to get ordered song list (even if features file exists)...")
        songs_bd_instance = BD('./MER_audio_taffc_dataset')
        if not songs_bd_instance.ordered_song_paths: # Check if paths were determined
            print("Critical: BD instance could not determine ordered song paths. Halting.")
            return

    # Attempt to load the features to check if Alínea 2.1 was successful or if file is usable
    try:
        # This variable will hold min_vals, max_vals, and normalized_feature_vectors
        all_feature_data = BD.load_features_from_file(features_path)
        if all_feature_data.shape[1] != 190: # 190 is the expected number of features
             print(f"Warning: Expected 190 features, but found {all_feature_data.shape[1]} in '{features_path}'. Results may be incorrect.")
    except Exception as e:
        print(f"Error loading or validating '{features_path}': {e}. Please delete it and re-run to regenerate.")
        return
        
    # --- ALÍNEA 2.2: Implementação e Avaliação do Centróide Espectral ---
    print("\n--- Iniciando Alínea 2.2: Centróide Espectral ---")
    
    # Parâmetros para o centróide espectral
    sr_sc = 22050  # Taxa de amostragem consistente com Librosa e o enunciado
    frame_length_sc = 2048 # Equivalente a 92.88ms @ 22050Hz
    hop_length_sc = 512    # Equivalente a 23.22ms @ 22050Hz

    rmse_values_sc = []
    pearson_corr_values_sc = []
    
    # Usar um número limitado de músicas para o teste do centróide para poupar tempo, e.g., as primeiras 10 ou um subconjunto.
    # O enunciado menciona "900 músicas" para guardar os resultados das métricas de erro.
    # Para desenvolvimento/teste rápido, podemos usar um subconjunto.
    # Vamos usar os caminhos das músicas carregados pela BD para consistência.
    
    # Re-initialize BD to get song paths for SC evaluation if not done already or too large
    # This assumes MER_audio_taffc_dataset contains the audio files.
    # We need the actual audio data for SC calculation.
    
    # Create a new BD instance to get file paths if necessary, or reuse if available and small.
    # For now, let's assume we need to list some audio files for SC calculation.
    # A better approach would be to use the song objects from a BD instance if Alínea 2.1 ran.

    # Simplified: Iterate through a directory of audios for SC eval
    # This part should ideally use the file paths from the `songs_bd` object if it was created
    # and if it's not too large for memory. For now, direct listing for SC:
    
    audio_files_for_sc_eval_path = './MER_audio_taffc_dataset' # Corrected to the parent directory
    
    # Check if the directory exists
    if not os.path.isdir(audio_files_for_sc_eval_path):
        print(f"Warning: Directory for SC evaluation audio files not found: {audio_files_for_sc_eval_path}")
        print("Skipping Alínea 2.2.2 (SC comparison with Librosa).")
        # Create dummy results file as per 2.2.3
        dummy_sc_results = np.zeros((900, 2)) # Assuming 900 songs
        np.savetxt(
            'resultados_metricas_sc.csv',
            dummy_sc_results,
            delimiter=',',
            header='RMSE_SC,Pearson_Correlation_SC',
            comments='',
            fmt=['%.6f', '%.6f']
        )
        print("Created dummy 'resultados_metricas_sc.csv'.")

    else:
        audio_files_list_sc = [os.path.join(dp, f) for dp, dn, filenames in os.walk(audio_files_for_sc_eval_path) for f in filenames if f.endswith('.mp3')]
        
        if not audio_files_list_sc:
            print(f"No MP3 files found in {audio_files_for_sc_eval_path} for SC evaluation.")
        else:
            print(f"Evaluating SC for {len(audio_files_list_sc)} files ...")
            
            # Limit to 900 files if more are found, or process all if fewer
            # files_to_process_sc = audio_files_list_sc[:900] if len(audio_files_list_sc) > 900 else audio_files_list_sc
            # For development, let's process fewer to speed up, e.g., first 5.
            # User should change this for full run.
            files_to_process_sc = audio_files_list_sc[:] 
            print(f"Processing SC for the first {len(files_to_process_sc)} files for speed...")


            for audio_file_path in files_to_process_sc:
                try:
                    y_sc, sr_loaded = librosa.load(audio_file_path, sr=sr_sc, mono=True)
                    if sr_loaded != sr_sc:
                         print(f"Warning: SR mismatch for {audio_file_path}. Expected {sr_sc}, got {sr_loaded}.")

                    # Calcular o centróide espectral com a implementação manual
                    sc_manual_frames = []
                    for start in range(0, len(y_sc) - frame_length_sc + 1, hop_length_sc):
                        frame = y_sc[start : start + frame_length_sc]
                        if len(frame) == frame_length_sc:
                             sc_manual_frames.append(compute_spectral_centroid(frame, sr_sc, frame_length_sc))
                    sc_manual = np.array(sc_manual_frames)

                    # Calcular o centróide espectral usando librosa
                    sc_librosa_frames = librosa.feature.spectral_centroid(y=y_sc, sr=sr_sc, n_fft=frame_length_sc, hop_length=hop_length_sc)
                    sc_librosa = sc_librosa_frames.flatten() # Librosa returns (1, T)

                    # Alinhar resultados (Librosa SC often has a 2-frame offset relative to manual STFT)
                    if len(sc_librosa) > 2:
                        sc_librosa_aligned = sc_librosa[2:]
                    else:
                        sc_librosa_aligned = np.array([]) # Not enough frames from librosa

                    # Ensure sc_manual and sc_librosa_aligned have the same length for comparison
                    min_len = min(len(sc_manual), len(sc_librosa_aligned))
                    sc_manual_trimmed = sc_manual[:min_len]
                    sc_librosa_trimmed = sc_librosa_aligned[:min_len]
                    
                    if min_len > 1: # Need at least 2 points for Pearson correlation
                        rmse_sc = np.sqrt(np.mean((sc_manual_trimmed - sc_librosa_trimmed) ** 2))
                        pearson_corr_sc, _ = pearsonr(sc_manual_trimmed, sc_librosa_trimmed)
                        
                        # Handle NaNs from pearsonr if one vector is constant
                        if np.isnan(pearson_corr_sc):
                            pearson_corr_sc = 0.0 # Or 1.0 if perfect match expected for constant
                    elif min_len == 1 and sc_manual_trimmed[0] == sc_librosa_trimmed[0]:
                        rmse_sc = 0.0
                        pearson_corr_sc = 1.0 # Perfect match for single point
                    elif min_len > 0: # Single point, different values
                        rmse_sc = np.abs(sc_manual_trimmed[0] - sc_librosa_trimmed[0])
                        pearson_corr_sc = 0.0 # No correlation definable, or undefined
                    else: # No comparable frames
                        rmse_sc = np.nan 
                        pearson_corr_sc = np.nan

                    rmse_values_sc.append(rmse_sc)
                    pearson_corr_values_sc.append(pearson_corr_sc)

                except Exception as e_sc:
                    print(f"Error processing SC for {audio_file_path}: {e_sc}")
                    rmse_values_sc.append(np.nan)
                    pearson_corr_values_sc.append(np.nan)
            
            # Guardar resultados da avaliação do SC (conforme 2.2.3)
            # O enunciado pede 900 linhas. Se processamos menos, precisamos preencher.
            num_results = len(rmse_values_sc)
            results_array_sc = np.full((900, 2), np.nan) # Initialize with NaN
            
            # Fill with actual results
            actual_results = np.column_stack((rmse_values_sc, pearson_corr_values_sc))
            results_array_sc[:num_results, :] = actual_results[:num_results, :] # Fill what we have
            
            np.savetxt(
                'resultados_metricas_sc.csv', 
                results_array_sc,
                delimiter=',',
                header='RMSE_SC,Pearson_Correlation_SC',
                comments='',
                fmt=['%.6f', '%.6f']
            )
            print("Resultados da avaliação do Spectral Centroid salvos em 'resultados_metricas_sc.csv'")


    # --- ALÍNEA 3: Implementação de Métricas de Similaridade ---
    print("\n--- Iniciando Alínea 3: Métricas de Similaridade ---")
    
    # Carregar os dados de features_statistics.csv 
    # all_feature_data já foi carregado no início da main()
    try:
        min_vals_dataset = all_feature_data[0, :]
        max_vals_dataset = all_feature_data[1, :]
        # As estatísticas carregadas já estão normalizadas conforme common.py foi alterado
        normalized_stats_all_songs = all_feature_data[2:, :] 
    except IndexError:
        print(f"ERRO: O ficheiro {features_path} não parece conter os dados esperados (min, max, stats).")
        print("Verifique se common.py está correto e re-gere o ficheiro de features.")
        return
    except Exception as e_load_features:
         print(f"ERRO ao processar dados de {features_path}: {e_load_features}")
         return

    if normalized_stats_all_songs.shape[0] == 0:
        print(f"ERRO: Nenhuma estatística de música encontrada em {features_path}.")
        return
    if normalized_stats_all_songs.shape[1] != 190:
        print(f"AVISO: Esperava 190 features normalizadas, mas encontrei {normalized_stats_all_songs.shape[1]}. As recomendações podem estar incorretas.")


    # Obter a lista ordenada de NOMES de ficheiros de música (para mapear índices para nomes)
    # Esta lógica agora usa o método da classe BD para garantir consistência com a ordem de features_statistics.csv
    if songs_bd_instance is None:
        # This case should ideally not be hit if the logic above is correct
        print("Error: songs_bd_instance not initialized. Re-initializing to get song order.")
        songs_bd_instance = BD('./MER_audio_taffc_dataset')
        if not songs_bd_instance.ordered_song_paths:
             print("Critical: BD instance could not determine ordered song paths after re-init. Halting.")
             return

    song_filenames_ordered = songs_bd_instance.get_ordered_song_filenames()

    # Convert Path objects from songs_bd_instance.ordered_song_paths to full string paths for other uses if needed
    # For rankings, we just need the names. For query song filename, we need the full path to find its index.
    # ordered_full_song_paths = [str(p) for p in songs_bd_instance.ordered_song_paths]

    if not song_filenames_ordered:
        print(f"ERRO: Nenhum nome de ficheiro MP3 obtido da instância BD. Verifique a inicialização da BD e o dataset.")
        return
    
    # Check consistency between number of songs in features and number of song names
    if len(song_filenames_ordered) != normalized_stats_all_songs.shape[0]:
        print(f"AVISO: Número de nomes de ficheiros ({len(song_filenames_ordered)}) não corresponde ao número de registos de estatísticas ({normalized_stats_all_songs.shape[0]}). As recomendações podem estar desalinhadas.")
        # Potentially truncate or error out based on severity. For now, a warning.
        if not normalized_stats_all_songs.any(): 
             print("Abortando devido à falta de estatísticas.")
             return

    # 3.2.1. Para a query: Selecionar uma música query
    # We need to find the index of the query song (e.g., "MT0000414517.mp3") in our `song_filenames_ordered` list
    query_song_target_name = "MT0000414517.mp3"
    query_song_index = -1
    try:
        query_song_index = song_filenames_ordered.index(query_song_target_name)
    except ValueError:
        print(f"ERRO: A música query '{query_song_target_name}' não foi encontrada na lista de músicas ordenadas do dataset.")
        print(f"Verifique o nome do ficheiro e se está presente no ficheiro de metadados e no dataset.")
        print(f"Primeiros ficheiros na lista ordenada: {song_filenames_ordered[:5]}")
        return

    if query_song_index >= normalized_stats_all_songs.shape[0]: # Should be caught by ValueError mostly
        print(f"ERRO: Índice da música query ({query_song_index} para '{query_song_target_name}') fora dos limites ({normalized_stats_all_songs.shape[0]} músicas no dataset).")
        return
        
    query_song_normalized_stats = normalized_stats_all_songs[query_song_index]
    # query_song_filename now refers to just the name, e.g., "MT0000414517.mp3"
    query_song_filename_for_print = song_filenames_ordered[query_song_index] 
    print(f"Música Query: {query_song_filename_for_print}")

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
    np.savetxt('similarity_euclidean.csv', distances_euclidean.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_manhattan.csv', distances_manhattan.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_cosine.csv', distances_cosine.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    print("Matrizes de similaridade (vetores de distância) guardadas em .csv")

    # 3.3. Para a query, criar os 3 rankings de similaridade (top 10)
    recommendation_count = 10

    def save_ranking_csv(filename, song_names_all_dataset, sorted_indices, distances, num_recommendations):
        # sorted_indices[0] é a própria query. Queremos top 10 *outras* musicas.
        
        
        # Indices of query + top 10 recommendations
        ranked_indices_to_save = sorted_indices[:num_recommendations + 1] 
        
        with open(filename, 'w', encoding='utf-8') as f_rank:
            for i, idx in enumerate(ranked_indices_to_save):
                if idx < len(song_names_all_dataset):
                    song_name = song_names_all_dataset[idx] # Should be just the filename e.g. "MTxxxx.mp3"
                    distance_val = distances[idx]
                    f_rank.write(f"{song_name},{distance_val:.6f}\n")
                else:
                    print(f"Warning: Index {idx} out of bounds for song_names_all_dataset when saving {filename}")

    # Obter nomes de ficheiros para os rankings (apenas o nome do ficheiro, não o caminho completo)
    # song_filenames_ordered_for_ranking = [p.name for p in song_files_ordered] # Old way
    # song_filenames_ordered já contém os nomes corretos na ordem correta

    # Ranking Euclidiano
    sorted_indices_euclidean = np.argsort(distances_euclidean)
    save_ranking_csv('ranking_euclidean.csv', song_filenames_ordered, sorted_indices_euclidean, distances_euclidean, recommendation_count)
    recommended_indices_euclidean = sorted_indices_euclidean[1 : recommendation_count + 1] # Top 10 excluding query

    # Ranking Manhattan
    sorted_indices_manhattan = np.argsort(distances_manhattan)
    save_ranking_csv('ranking_manhattan.csv', song_filenames_ordered, sorted_indices_manhattan, distances_manhattan, recommendation_count)
    recommended_indices_manhattan = sorted_indices_manhattan[1 : recommendation_count + 1] # Top 10 excluding query

    # Ranking Cosseno
    sorted_indices_cosine = np.argsort(distances_cosine)
    save_ranking_csv('ranking_cosine.csv', song_filenames_ordered, sorted_indices_cosine, distances_cosine, recommendation_count)
    recommended_indices_cosine = sorted_indices_cosine[1 : recommendation_count + 1] # Top 10 excluding query
    
    print("Rankings (top 10 + query) guardados em ranking_*.csv")


    print(f"\nTop {recommendation_count} Recomendações para {query_song_filename_for_print} (excluindo a própria música):")
    
    # Para formatação igual ao ficheiro de validação
    # Guardar nomes e distâncias para impressão posterior no formato do ficheiro de validação
    
    print("\nRecomendações - Distância Euclidiana:")
    recommended_filenames_euclidean = []
    recommended_distances_euclidean_values = []
    # Incluir a própria música query para corresponder ao formato de saida do ficheiro de validação
    # A query song é sorted_indices_euclidean[0]
    if sorted_indices_euclidean[0] < len(song_filenames_ordered):
        recommended_filenames_euclidean.append(song_filenames_ordered[sorted_indices_euclidean[0]])
        recommended_distances_euclidean_values.append(distances_euclidean[sorted_indices_euclidean[0]])
        
    for idx in recommended_indices_euclidean: # Top 10, excluindo query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_euclidean)}. {song_filenames_ordered[idx]} (Distância: {distances_euclidean[idx]:.4f})")
            recommended_filenames_euclidean.append(song_filenames_ordered[idx])
            recommended_distances_euclidean_values.append(distances_euclidean[idx])
        else:
            print(f"{len(recommended_filenames_euclidean)}. Índice {idx} fora dos limites para nomes de ficheiro.")

    print("\nRecomendações - Distância Manhattan:")
    recommended_filenames_manhattan = []
    recommended_distances_manhattan_values = []
    if sorted_indices_manhattan[0] < len(song_filenames_ordered):
        recommended_filenames_manhattan.append(song_filenames_ordered[sorted_indices_manhattan[0]])
        recommended_distances_manhattan_values.append(distances_manhattan[sorted_indices_manhattan[0]])

    for idx in recommended_indices_manhattan: # Top 10, excluindo query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_manhattan)}. {song_filenames_ordered[idx]} (Distância: {distances_manhattan[idx]:.4f})")
            recommended_filenames_manhattan.append(song_filenames_ordered[idx])
            recommended_distances_manhattan_values.append(distances_manhattan[idx])
        else:
            print(f"{len(recommended_filenames_manhattan)}. Índice {idx} fora dos limites para nomes de ficheiro.")
            
    print("\nRecomendações - Distância Cosseno:")
    recommended_filenames_cosine = []
    recommended_distances_cosine_values = []
    if sorted_indices_cosine[0] < len(song_filenames_ordered):
        recommended_filenames_cosine.append(song_filenames_ordered[sorted_indices_cosine[0]])
        recommended_distances_cosine_values.append(distances_cosine[sorted_indices_cosine[0]])

    for idx in recommended_indices_cosine: # Top 10, excluindo query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_cosine)}. {song_filenames_ordered[idx]} (Distância: {distances_cosine[idx]:.4f})")
            recommended_filenames_cosine.append(song_filenames_ordered[idx])
            recommended_distances_cosine_values.append(distances_cosine[idx])
        else:
            print(f"{len(recommended_filenames_cosine)}. Índice {idx} fora dos limites para nomes de ficheiro.")
            
    print("\n--- Alínea 3 Concluída ---")

    # --- ALÍNEA 4.1: Avaliação Objectiva ---
    print("\n--- Iniciando Alínea 4.1: Avaliação Objectiva ---")

    metadata_file_path = 'MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    try:
        metadata_df = pd.read_csv(metadata_file_path)
    except FileNotFoundError:
        print(f"ERRO: Ficheiro de metadados '{metadata_file_path}' não encontrado.")
        return

    # Certificar que a coluna 'Song' existe (filename sem .mp3)
    if 'Song' not in metadata_df.columns:
        print(f"ERRO: Coluna 'Song' não encontrada em '{metadata_file_path}'.")
        return
    
    # Indexar o DataFrame pelo ID da música para lookup fácil
    metadata_df = metadata_df.set_index('Song')

    # 4.1.1 Obter metadados da música query
    query_song_id_stem = Path(query_song_filename_for_print).stem
    if query_song_id_stem not in metadata_df.index:
        print(f"ERRO: Metadados para a música query '{query_song_filename_for_print}' (ID: {query_song_id_stem}) não encontrados no ficheiro CSV.")
        return
    query_metadata = metadata_df.loc[query_song_id_stem]

    def get_metadata_similarity_score(query_meta, target_meta):
        score = 0
        
        # Comparar Artista
        if pd.notna(query_meta['Artist']) and pd.notna(target_meta['Artist']) and query_meta['Artist'] == target_meta['Artist']:
            score += 1
            
        # Comparar Géneros (MoodsStrSplit e GenresStr)
        def compare_split_str(query_str, target_str):
            if pd.isna(query_str) or pd.isna(target_str) or not query_str.strip() or not target_str.strip():
                return 0
            query_items = set(item.strip().lower() for item in query_str.split(';') if item.strip())
            target_items = set(item.strip().lower() for item in target_str.split(';') if item.strip())
            return len(query_items.intersection(target_items))

        score += compare_split_str(query_meta.get('GenresStr', ''), target_meta.get('GenresStr', ''))
        score += compare_split_str(query_meta.get('MoodsStrSplit', ''), target_meta.get('MoodsStrSplit', ''))
        
        return score

    metadata_similarity_scores = np.zeros(num_songs_in_db)
    for i in range(num_songs_in_db):
        target_song_id_stem = Path(song_filenames_ordered[i]).stem
        if target_song_id_stem in metadata_df.index:
            target_metadata = metadata_df.loc[target_song_id_stem]
            metadata_similarity_scores[i] = get_metadata_similarity_score(query_metadata, target_metadata)
        else:
            # print(f"AVISO: Metadados para '{song_filenames_ordered[i]}' não encontrados. Score será 0.")
            metadata_similarity_scores[i] = 0 # Ou outra estratégia

    # Guardar a matriz de similaridade baseada em contexto
    np.savetxt('similarity_metadata.csv', metadata_similarity_scores.reshape(-1, 1), delimiter=',', comments='', fmt='%d')
    print("Matriz de similaridade por metadados guardada em 'similarity_metadata.csv'")

    # Obter ranking por metadados (top 10 + query, ou top 11 para display)
    # Scores são "maior é melhor"
    sorted_indices_metadata = np.argsort(metadata_similarity_scores)[::-1] 
    
    # Para display, seguindo o formato do ficheiro validação (parece ter 11 musicas)
    num_recommendations_display_metadata = recommendation_count + 1 
    recommended_indices_metadata_display = sorted_indices_metadata[:num_recommendations_display_metadata]
    
    recommended_filenames_metadata = [song_filenames_ordered[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]
    recommended_scores_metadata = [metadata_similarity_scores[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]

    # Para cálculo da precisão, consideramos as top 10 *outras* músicas
    # (excluindo a própria query song que estaria em sorted_indices_metadata[0])
    # Se a query song não for a primeira, esta lógica precisa de ajuste, mas é esperado que seja.
    if sorted_indices_metadata[0] == query_song_index:
         relevant_indices_for_precision_metadata = sorted_indices_metadata[1 : recommendation_count + 1]
    else:
        # Caso a query não seja a mais similar a si mesma (improvável com esta métrica, mas por segurança)
        # Removemos a query e pegamos as top 10 restantes
        temp_sorted_metadata = [idx for idx in sorted_indices_metadata if idx != query_song_index]
        relevant_indices_for_precision_metadata = temp_sorted_metadata[:recommendation_count]

    relevant_set_for_precision = set(relevant_indices_for_precision_metadata)

    # 4.1.2 Calcular a métrica precision
    # recommended_indices_euclidean, etc., já são os top 10 excluindo a query
    
    precision_euclidean = len(set(recommended_indices_euclidean).intersection(relevant_set_for_precision)) / recommendation_count
    precision_manhattan = len(set(recommended_indices_manhattan).intersection(relevant_set_for_precision)) / recommendation_count
    precision_cosine = len(set(recommended_indices_cosine).intersection(relevant_set_for_precision)) / recommendation_count

    # Apresentar resultados no formato do ficheiro de validação
    output_txt_filename = "rankings_and_precision_output.txt"
    with open(output_txt_filename, 'w', encoding='utf-8') as f_out:
        print("\n\n- Rankings:", file=f_out)
        print("\nRanking: Euclidean-------------", file=f_out)
        print(f"{recommended_filenames_euclidean}", file=f_out)
        print(f"{np.array(recommended_distances_euclidean_values)}", file=f_out)
        
        print("\nRanking: Manhattan-------------", file=f_out)
        print(f"{recommended_filenames_manhattan}", file=f_out)
        print(f"{np.array(recommended_distances_manhattan_values)}", file=f_out)

        print("\nRanking: Cosine-------------", file=f_out)
        print(f"{recommended_filenames_cosine}", file=f_out)
        print(f"{np.array(recommended_distances_cosine_values)}", file=f_out)
        
        print("\nRanking: Metadata-------------", file=f_out)
        print(f"{recommended_filenames_metadata}", file=f_out)
        # Convert scores to int for printing, as in the example file
        print(f"{np.array(recommended_scores_metadata, dtype=int)}", file=f_out)

        print(f"\nPrecision de:  {precision_euclidean * 100:.1f}", file=f_out)
        print(f"Precision dm:  {precision_manhattan * 100:.1f}", file=f_out)
        print(f"Precision dc:  {precision_cosine * 100:.1f}", file=f_out)

    print(f"\nO relatório de rankings e precisão foi guardado em: {output_txt_filename}")
    print("\n--- Alínea 4.1 Concluída ---")

if __name__ == '__main__':
    main()
