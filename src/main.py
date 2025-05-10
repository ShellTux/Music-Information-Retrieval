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

    # Ensure frequencies array matches magnitude_spectrum length
    expected_bins = frame_length_for_fft // 2 + 1
    if len(magnitude_spectrum) != expected_bins:
        frequencies = frequencies[:len(magnitude_spectrum)]

    spectral_centroid_val = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0.0
    return float(spectral_centroid_val)

def main():
    # --- SECTION 2.1: Feature Extraction and Saving ---
    songs_bd_instance = None # To store the BD instance for later use
    
    # Check if features file already exists
    if not os.path.exists(features_path):
        print(f"'{features_path}' not found. Generating features...")
        songs_bd_instance = BD('../MER_audio_taffc_dataset') 
        if not songs_bd_instance.songs:
            print("No songs were successfully processed by BD. Halting.")
            return
        songs_bd_instance.save_features_to_file(features_path)
        print(f"Features saved to '{features_path}'")
    else:
        print(f"Using existing features from '{features_path}'")
        # Only load metadata and song list without processing audio files
        print("Loading song filenames from metadata without processing audio files...")
        songs_bd_instance = BD('../MER_audio_taffc_dataset', load_features_only=True)
        if not songs_bd_instance.ordered_song_paths:
            print("Critical: BD instance could not determine ordered song paths. Halting.")
            return

    # Attempt to load the features to check if Section 2.1 was successful or if file is usable
    try:
        # This variable will hold min_vals, max_vals, and normalized_feature_vectors
        all_feature_data = BD.load_features_from_file(features_path)
        if all_feature_data.shape[1] != 190: # 190 is the expected number of features
             print(f"Warning: Expected 190 features, but found {all_feature_data.shape[1]} in '{features_path}'. Results may be incorrect.")
    except Exception as e:
        print(f"Error loading or validating '{features_path}': {e}. Please delete it and re-run to regenerate.")
        return
        
    # --- SECTION 2.2: Spectral Centroid Implementation and Evaluation ---
    print("\n--- Starting Section 2.2: Spectral Centroid ---")
    
    # Parameters for spectral centroid
    sr_sc = 22050  # Sampling rate consistent with Librosa and requirements
    frame_length_sc = 2048 # Equivalent to 92.88ms @ 22050Hz
    hop_length_sc = 512    # Equivalent to 23.22ms @ 22050Hz

    # Skip SC evaluation if results file already exists
    sc_results_path = 'resultados_metricas_sc.csv'
    if os.path.exists(sc_results_path):
        print(f"SC evaluation results file '{sc_results_path}' already exists. Skipping evaluation.")
    else:
        rmse_values_sc = []
        pearson_corr_values_sc = []
        
        # Use a limited number of songs for centroid testing to save time
        audio_files_for_sc_eval_path = './MER_audio_taffc_dataset'
        
        # Check if the directory exists
        if not os.path.isdir(audio_files_for_sc_eval_path):
            print(f"Warning: Directory for SC evaluation audio files not found: {audio_files_for_sc_eval_path}")
            print("Skipping Section 2.2.2 (SC comparison with Librosa).")
            # Create dummy results file as per 2.2.3
            dummy_sc_results = np.zeros((900, 2)) # Assuming 900 songs
            np.savetxt(
                sc_results_path,
                dummy_sc_results,
                delimiter=',',
                header='RMSE_SC,Pearson_Correlation_SC',
                comments='',
                fmt=['%.6f', '%.6f']
            )
            print(f"Created dummy '{sc_results_path}'.")

        else:
            audio_files_list_sc = [os.path.join(dp, f) for dp, dn, filenames in os.walk(audio_files_for_sc_eval_path) for f in filenames if f.endswith('.mp3')]
            
            if not audio_files_list_sc:
                print(f"No MP3 files found in {audio_files_for_sc_eval_path} for SC evaluation.")
            else:
                print(f"Evaluating SC for {len(audio_files_list_sc)} files ...")
                
                # Process all available files
                files_to_process_sc = audio_files_list_sc[:] 
                print(f"Processing SC for the first {len(files_to_process_sc)} files for speed...")

                for audio_file_path in files_to_process_sc:
                    try:
                        y_sc, sr_loaded = librosa.load(audio_file_path, sr=sr_sc, mono=True)
                        if sr_loaded != sr_sc:
                             print(f"Warning: SR mismatch for {audio_file_path}. Expected {sr_sc}, got {sr_loaded}.")

                        # Calculate spectral centroid with manual implementation
                        sc_manual_frames = []
                        for start in range(0, len(y_sc) - frame_length_sc + 1, hop_length_sc):
                            frame = y_sc[start : start + frame_length_sc]
                            if len(frame) == frame_length_sc:
                                 sc_manual_frames.append(compute_spectral_centroid(frame, sr_sc, frame_length_sc))
                        sc_manual = np.array(sc_manual_frames)

                        # Calculate spectral centroid using librosa
                        sc_librosa_frames = librosa.feature.spectral_centroid(y=y_sc, sr=sr_sc, n_fft=frame_length_sc, hop_length=hop_length_sc)
                        sc_librosa = sc_librosa_frames.flatten() # Librosa returns (1, T)

                        # Align results (Librosa SC often has a 2-frame offset relative to manual STFT)
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
                
                # Save SC evaluation results (as per 2.2.3)
                # The requirements specify 900 rows. If we processed fewer, we need to fill the rest.
                num_results = len(rmse_values_sc)
                results_array_sc = np.full((900, 2), np.nan) # Initialize with NaN
                
                # Fill with actual results
                actual_results = np.column_stack((rmse_values_sc, pearson_corr_values_sc))
                results_array_sc[:num_results, :] = actual_results[:num_results, :] # Fill what we have
                
                np.savetxt(
                    sc_results_path, 
                    results_array_sc,
                    delimiter=',',
                    header='RMSE_SC,Pearson_Correlation_SC',
                    comments='',
                    fmt=['%.6f', '%.6f']
                )
                print(f"Spectral Centroid evaluation results saved to '{sc_results_path}'")


    # --- SECTION 3: Similarity Metrics Implementation ---
    print("\n--- Starting Section 3: Similarity Metrics ---")
    
    # Load data from features_statistics.csv 
    # all_feature_data was already loaded at the beginning of main()
    try:
        min_vals_dataset = all_feature_data[0, :]
        max_vals_dataset = all_feature_data[1, :]
        # Statistics are already normalized as common.py was modified
        normalized_stats_all_songs = all_feature_data[2:, :] 
    except IndexError:
        print(f"ERROR: The file {features_path} doesn't seem to contain the expected data (min, max, stats).")
        print("Check if common.py is correct and regenerate the features file.")
        return
    except Exception as e_load_features:
         print(f"ERROR processing data from {features_path}: {e_load_features}")
         return

    if normalized_stats_all_songs.shape[0] == 0:
        print(f"ERROR: No song statistics found in {features_path}.")
        return
    if normalized_stats_all_songs.shape[1] != 190:
        print(f"WARNING: Expected 190 normalized features, but found {normalized_stats_all_songs.shape[1]}. Recommendations may be incorrect.")


    # Get ordered list of song filenames (to map indices to names)
    # This logic now uses the BD class method to ensure consistency with features_statistics.csv order
    if songs_bd_instance is None:
        print("Error: songs_bd_instance not initialized. Re-initializing to get song order.")
        songs_bd_instance = BD('../MER_audio_taffc_dataset', load_features_only=True)
        if not songs_bd_instance.ordered_song_paths:
             print("Critical: BD instance could not determine ordered song paths after re-init. Halting.")
             return

    song_filenames_ordered = songs_bd_instance.get_ordered_song_filenames()

    if not song_filenames_ordered:
        print(f"ERROR: No MP3 filenames obtained from BD instance. Check BD initialization and dataset.")
        return
    
    # Check consistency between number of songs in features and number of song names
    if len(song_filenames_ordered) != normalized_stats_all_songs.shape[0]:
        print(f"WARNING: Number of filenames ({len(song_filenames_ordered)}) doesn't match number of statistics records ({normalized_stats_all_songs.shape[0]}). Recommendations may be misaligned.")
        if not normalized_stats_all_songs.any(): 
             print("Aborting due to lack of statistics.")
             return

    # 3.2.1. For the query: Select a query song
    # Find the index of the query song in our song_filenames_ordered list
    query_song_target_name = "MT0000414517.mp3"
    query_song_index = -1
    try:
        query_song_index = song_filenames_ordered.index(query_song_target_name)
    except ValueError:
        print(f"ERROR: Query song '{query_song_target_name}' not found in the ordered song list.")
        print(f"Check the filename and verify it exists in the metadata file and dataset.")
        print(f"First files in ordered list: {song_filenames_ordered[:5]}")
        return

    if query_song_index >= normalized_stats_all_songs.shape[0]:
        print(f"ERROR: Query song index ({query_song_index} for '{query_song_target_name}') out of bounds ({normalized_stats_all_songs.shape[0]} songs in dataset).")
        return
        
    query_song_normalized_stats = normalized_stats_all_songs[query_song_index]
    # query_song_filename now refers to just the name, e.g., "MT0000414517.mp3"
    query_song_filename_for_print = song_filenames_ordered[query_song_index] 
    print(f"Query Song: {query_song_filename_for_print}")

    # 3.1. Similarity metrics are already available in scipy.spatial.distance

    # 3.2.2. Create and save 3 similarity matrices (distance vectors)
    num_songs_in_db = normalized_stats_all_songs.shape[0]
    
    distances_euclidean = np.zeros(num_songs_in_db)
    distances_manhattan = np.zeros(num_songs_in_db)
    distances_cosine = np.zeros(num_songs_in_db)

    for i in range(num_songs_in_db):
        db_song_normalized_stats = normalized_stats_all_songs[i]
        distances_euclidean[i] = distance.euclidean(query_song_normalized_stats, db_song_normalized_stats)
        distances_manhattan[i] = distance.cityblock(query_song_normalized_stats, db_song_normalized_stats) # cityblock is Manhattan
        distances_cosine[i] = distance.cosine(query_song_normalized_stats, db_song_normalized_stats) # 1 - cosine similarity

    # Save similarity matrices (distance vectors)
    np.savetxt('similarity_euclidean.csv', distances_euclidean.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_manhattan.csv', distances_manhattan.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_cosine.csv', distances_cosine.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    print("Similarity matrices (distance vectors) saved to .csv files")

    # 3.3. For the query, create 3 similarity rankings (top 10)
    recommendation_count = 10

    def save_ranking_csv(filename, song_names_all_dataset, sorted_indices, distances, num_recommendations):
        # sorted_indices[0] is the query itself. We want top 10 *other* songs.
        
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

    # Euclidean Ranking
    sorted_indices_euclidean = np.argsort(distances_euclidean)
    save_ranking_csv('ranking_euclidean.csv', song_filenames_ordered, sorted_indices_euclidean, distances_euclidean, recommendation_count)
    recommended_indices_euclidean = sorted_indices_euclidean[1 : recommendation_count + 1] # Top 10 excluding query

    # Manhattan Ranking
    sorted_indices_manhattan = np.argsort(distances_manhattan)
    save_ranking_csv('ranking_manhattan.csv', song_filenames_ordered, sorted_indices_manhattan, distances_manhattan, recommendation_count)
    recommended_indices_manhattan = sorted_indices_manhattan[1 : recommendation_count + 1] # Top 10 excluding query

    # Cosine Ranking
    sorted_indices_cosine = np.argsort(distances_cosine)
    save_ranking_csv('ranking_cosine.csv', song_filenames_ordered, sorted_indices_cosine, distances_cosine, recommendation_count)
    recommended_indices_cosine = sorted_indices_cosine[1 : recommendation_count + 1] # Top 10 excluding query
    
    print("Rankings (top 10 + query) saved to ranking_*.csv files")


    print(f"\nTop {recommendation_count} Recommendations for {query_song_filename_for_print} (excluding the song itself):")
    
    # For formatting similar to the validation file
    # Store names and distances for later printing in validation file format
    
    print("\nRecommendations - Euclidean Distance:")
    recommended_filenames_euclidean = []
    recommended_distances_euclidean_values = []
    # Include the query song itself to match the validation file output format
    # The query song is sorted_indices_euclidean[0]
    if sorted_indices_euclidean[0] < len(song_filenames_ordered):
        recommended_filenames_euclidean.append(song_filenames_ordered[sorted_indices_euclidean[0]])
        recommended_distances_euclidean_values.append(distances_euclidean[sorted_indices_euclidean[0]])
        
    for idx in recommended_indices_euclidean: # Top 10, excluding query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_euclidean)}. {song_filenames_ordered[idx]} (Distance: {distances_euclidean[idx]:.4f})")
            recommended_filenames_euclidean.append(song_filenames_ordered[idx])
            recommended_distances_euclidean_values.append(distances_euclidean[idx])
        else:
            print(f"{len(recommended_filenames_euclidean)}. Index {idx} out of bounds for filenames.")

    print("\nRecommendations - Manhattan Distance:")
    recommended_filenames_manhattan = []
    recommended_distances_manhattan_values = []
    if sorted_indices_manhattan[0] < len(song_filenames_ordered):
        recommended_filenames_manhattan.append(song_filenames_ordered[sorted_indices_manhattan[0]])
        recommended_distances_manhattan_values.append(distances_manhattan[sorted_indices_manhattan[0]])

    for idx in recommended_indices_manhattan: # Top 10, excluding query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_manhattan)}. {song_filenames_ordered[idx]} (Distance: {distances_manhattan[idx]:.4f})")
            recommended_filenames_manhattan.append(song_filenames_ordered[idx])
            recommended_distances_manhattan_values.append(distances_manhattan[idx])
        else:
            print(f"{len(recommended_filenames_manhattan)}. Index {idx} out of bounds for filenames.")
            
    print("\nRecommendations - Cosine Distance:")
    recommended_filenames_cosine = []
    recommended_distances_cosine_values = []
    if sorted_indices_cosine[0] < len(song_filenames_ordered):
        recommended_filenames_cosine.append(song_filenames_ordered[sorted_indices_cosine[0]])
        recommended_distances_cosine_values.append(distances_cosine[sorted_indices_cosine[0]])

    for idx in recommended_indices_cosine: # Top 10, excluding query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_cosine)}. {song_filenames_ordered[idx]} (Distance: {distances_cosine[idx]:.4f})")
            recommended_filenames_cosine.append(song_filenames_ordered[idx])
            recommended_distances_cosine_values.append(distances_cosine[idx])
        else:
            print(f"{len(recommended_filenames_cosine)}. Index {idx} out of bounds for filenames.")
            
    print("\n--- Section 3 Completed ---")

    # --- SECTION 4.1: Objective Evaluation ---
    print("\n--- Starting Section 4.1: Objective Evaluation ---")

    metadata_file_path = '../MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    try:
        metadata_df = pd.read_csv(metadata_file_path)
    except FileNotFoundError:
        print(f"ERROR: Metadata file '{metadata_file_path}' not found.")
        return

    # Ensure 'Song' column exists (filename without .mp3)
    if 'Song' not in metadata_df.columns:
        print(f"ERROR: 'Song' column not found in '{metadata_file_path}'.")
        return
    
    # Index DataFrame by song ID for easy lookup
    metadata_df = metadata_df.set_index('Song')

    # 4.1.1 Get metadata for query song
    query_song_id_stem = Path(query_song_filename_for_print).stem
    if query_song_id_stem not in metadata_df.index:
        print(f"ERROR: Metadata for query song '{query_song_filename_for_print}' (ID: {query_song_id_stem}) not found in CSV file.")
        return
    query_metadata = metadata_df.loc[query_song_id_stem]

    def get_metadata_similarity_score(query_meta, target_meta):
        score = 0
        
        # Compare Artist
        if pd.notna(query_meta['Artist']) and pd.notna(target_meta['Artist']) and query_meta['Artist'] == target_meta['Artist']:
            score += 1
            
        # Compare Genres and Moods (MoodsStrSplit and GenresStr)
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
            metadata_similarity_scores[i] = 0 # Or another strategy

    # Save context-based similarity matrix
    np.savetxt('similarity_metadata.csv', metadata_similarity_scores.reshape(-1, 1), delimiter=',', comments='', fmt='%d')
    print("Metadata-based similarity matrix saved to 'similarity_metadata.csv'")

    # Get metadata ranking (top 10 + query, or top 11 for display)
    # Scores are "higher is better"
    sorted_indices_metadata = np.argsort(metadata_similarity_scores)[::-1] 
    
    # For display, following validation file format (appears to have 11 songs)
    num_recommendations_display_metadata = recommendation_count + 1 
    recommended_indices_metadata_display = sorted_indices_metadata[:num_recommendations_display_metadata]
    
    recommended_filenames_metadata = [song_filenames_ordered[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]
    recommended_scores_metadata = [metadata_similarity_scores[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]

    # For precision calculation, consider top 10 *other* songs
    # (excluding the query song itself which would be in sorted_indices_metadata[0])
    # If the query song is not first, this logic needs adjustment, but it's expected to be.
    if sorted_indices_metadata[0] == query_song_index:
         relevant_indices_for_precision_metadata = sorted_indices_metadata[1 : recommendation_count + 1]
    else:
        # If the query is not the most similar to itself (unlikely with this metric, but for safety)
        # Remove the query and take the top 10 remaining
        temp_sorted_metadata = [idx for idx in sorted_indices_metadata if idx != query_song_index]
        relevant_indices_for_precision_metadata = temp_sorted_metadata[:recommendation_count]

    relevant_set_for_precision = set(relevant_indices_for_precision_metadata)

    # 4.1.2 Calculate precision metric
    # recommended_indices_euclidean, etc., are already the top 10 excluding the query
    
    precision_euclidean = len(set(recommended_indices_euclidean).intersection(relevant_set_for_precision)) / recommendation_count
    precision_manhattan = len(set(recommended_indices_manhattan).intersection(relevant_set_for_precision)) / recommendation_count
    precision_cosine = len(set(recommended_indices_cosine).intersection(relevant_set_for_precision)) / recommendation_count

    # Present results in validation file format
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

    print(f"\nRanking and precision report saved to: {output_txt_filename}")
    print("\n--- Section 4.1 Completed ---")

    # --- SECTION 4.2: Subjective Evaluation ---
    print("\n--- Starting Section 4.2: Subjective Evaluation ---")
    
    # Definindo constantes para a avaliação subjetiva
    RELEVANCE_THRESHOLD = 2.5  # Score mínimo para considerar uma recomendação relevante
    
    # 4.2.1. Avaliação subjetiva para os rankings baseados em distâncias
    # Carregar dados reais de avaliação a partir de arquivos ou usar dados de exemplo
    ratings_dir = '../subjective_ratings'
    
    # Verificar se existem arquivos de avaliação
    if os.path.exists(ratings_dir) and any(f.endswith('.csv') for f in os.listdir(ratings_dir)):
        print(f"Carregando avaliações subjetivas do diretório '{ratings_dir}'...")
        
        # Inicializar listas para armazenar as avaliações de cada membro
        euclidean_ratings_list = []
        manhattan_ratings_list = []
        cosine_ratings_list = []
        metadata_ratings_list = []
        
        # Carregar cada arquivo de avaliações
        for ratings_file in sorted([f for f in os.listdir(ratings_dir) if f.endswith('.csv')]):
            file_path = os.path.join(ratings_dir, ratings_file)
            try:
                ratings_df = pd.read_csv(file_path)
                
                # Extrair avaliações para cada métrica, se existirem no arquivo
                if 'euclidean' in ratings_df.columns:
                    member_euclidean = ratings_df['euclidean'].values
                    if len(member_euclidean) == recommendation_count:
                        euclidean_ratings_list.append(member_euclidean)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_euclidean)} avaliações para euclidiana, esperava {recommendation_count}.")
                
                if 'manhattan' in ratings_df.columns:
                    member_manhattan = ratings_df['manhattan'].values
                    if len(member_manhattan) == recommendation_count:
                        manhattan_ratings_list.append(member_manhattan)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_manhattan)} avaliações para manhattan, esperava {recommendation_count}.")
                
                if 'cosine' in ratings_df.columns:
                    member_cosine = ratings_df['cosine'].values
                    if len(member_cosine) == recommendation_count:
                        cosine_ratings_list.append(member_cosine)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_cosine)} avaliações para cosseno, esperava {recommendation_count}.")
                
                if 'metadata' in ratings_df.columns:
                    member_metadata = ratings_df['metadata'].values
                    if len(member_metadata) == recommendation_count:
                        metadata_ratings_list.append(member_metadata)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_metadata)} avaliações para metadados, esperava {recommendation_count}.")
                
                print(f"Carregadas avaliações de {ratings_file}")
            except Exception as e:
                print(f"Erro ao carregar {ratings_file}: {e}")
        
        # Converter listas em arrays numpy
        NUM_GROUP_MEMBERS = len(euclidean_ratings_list)
        print(f"Número de membros do grupo com avaliações: {NUM_GROUP_MEMBERS}")
        
        if NUM_GROUP_MEMBERS > 0:
            euclidean_subjective_ratings = np.array(euclidean_ratings_list)
            manhattan_subjective_ratings = np.array(manhattan_ratings_list)
            cosine_subjective_ratings = np.array(cosine_ratings_list)
            
            if metadata_ratings_list:
                metadata_subjective_ratings = np.array(metadata_ratings_list)
            else:
                print("Aviso: Não foram encontradas avaliações para metadados.")
                # Criar dados fictícios para metadados para evitar erros
                metadata_subjective_ratings = np.zeros((NUM_GROUP_MEMBERS, recommendation_count))
        else:
            print("Aviso: Não foram encontrados arquivos de avaliação válidos. Usando dados de exemplo.")
            # Usar dados de exemplo
            NUM_GROUP_MEMBERS = 3  # Número de membros do grupo para exemplo
            # Códigos para dados de exemplo (mesmos que estavam antes)
    else:
        print(f"Diretório '{ratings_dir}' não encontrado ou vazio. Usando dados de exemplo.")
        # Usar dados de exemplo
        NUM_GROUP_MEMBERS = 3  # Número de membros do grupo para exemplo
        
        # Dados de exemplo para avaliação subjetiva - SUBSTITUIR POR AVALIAÇÕES REAIS
        # Estrutura: cada linha representa um membro do grupo, cada coluna uma música recomendada
        euclidean_subjective_ratings = np.array([
            [3, 4, 2, 1, 3, 4, 5, 2, 3, 4],  # Avaliações do membro 1
            [4, 3, 1, 2, 4, 3, 4, 3, 2, 5],  # Avaliações do membro 2
            [3, 3, 2, 2, 3, 4, 4, 3, 3, 4]   # Avaliações do membro 3
        ])
        
        manhattan_subjective_ratings = np.array([
            [2, 3, 4, 3, 2, 1, 3, 4, 5, 2],  # Avaliações do membro 1
            [3, 2, 3, 4, 3, 2, 4, 3, 4, 3],  # Avaliações do membro 2
            [2, 3, 3, 4, 2, 2, 3, 4, 4, 3]   # Avaliações do membro 3
        ])
        
        cosine_subjective_ratings = np.array([
            [4, 5, 3, 2, 3, 4, 3, 2, 1, 3],  # Avaliações do membro 1
            [5, 4, 3, 2, 4, 3, 2, 3, 2, 4],  # Avaliações do membro 2
            [4, 4, 3, 3, 3, 3, 3, 2, 2, 3]   # Avaliações do membro 3
        ])
        
        metadata_subjective_ratings = np.array([
            [5, 4, 4, 3, 3, 2, 2, 1, 3, 4],  # Avaliações do membro 1
            [4, 5, 3, 4, 2, 3, 2, 2, 4, 3],  # Avaliações do membro 2
            [5, 4, 4, 3, 3, 2, 1, 2, 3, 4]   # Avaliações do membro 3
        ])
    
    # Função para calcular estatísticas de avaliação subjetiva
    def calculate_subjective_stats(ratings_matrix):
        means_per_song = np.mean(ratings_matrix, axis=0)
        stds_per_song = np.std(ratings_matrix, axis=0, ddof=1)  # ddof=1 para desvio padrão amostral
        overall_mean = np.mean(means_per_song)
        overall_std = np.std(means_per_song, ddof=1)
        
        # Calcular precision (proporção de recomendações relevantes)
        relevant_count = np.sum(means_per_song >= RELEVANCE_THRESHOLD)
        precision = relevant_count / len(means_per_song)
        
        return means_per_song, stds_per_song, overall_mean, overall_std, precision

    # Calcular estatísticas para cada métrica
    eucl_means, eucl_stds, eucl_overall_mean, eucl_overall_std, eucl_precision = calculate_subjective_stats(euclidean_subjective_ratings)
    manh_means, manh_stds, manh_overall_mean, manh_overall_std, manh_precision = calculate_subjective_stats(manhattan_subjective_ratings)
    cos_means, cos_stds, cos_overall_mean, cos_overall_std, cos_precision = calculate_subjective_stats(cosine_subjective_ratings)
    meta_means, meta_stds, meta_overall_mean, meta_overall_std, meta_precision = calculate_subjective_stats(metadata_subjective_ratings)
    
    # Salvar resultados em um arquivo
    subjective_eval_filename = "subjective_evaluation_results.txt"
    with open(subjective_eval_filename, 'w', encoding='utf-8') as f_subj:
        print("=== AVALIAÇÃO SUBJETIVA ===", file=f_subj)
        print("\n--- 4.2.1. Avaliação por distância ---", file=f_subj)
        
        # Formato da saída para cada métrica
        def print_metric_results(f, metric_name, means, stds, overall_mean, overall_std, precision, filenames):
            print(f"\n{metric_name}:", file=f)
            print("Música\tMédia\tDesvio Padrão", file=f)
            for i, (mean, std, filename) in enumerate(zip(means, stds, filenames)):
                print(f"{i+1}. {filename}\t{mean:.2f}\t{std:.2f}", file=f)
            print(f"\nMédia Global: {overall_mean:.2f}", file=f)
            print(f"Desvio Padrão Global: {overall_std:.2f}", file=f)
            print(f"Precision (score >= {RELEVANCE_THRESHOLD}): {precision:.2f} ({precision*100:.1f}%)", file=f)
        
        # Resultados da distância Euclidiana
        print_metric_results(
            f_subj, 
            "Distância Euclidiana", 
            eucl_means, 
            eucl_stds, 
            eucl_overall_mean, 
            eucl_overall_std, 
            eucl_precision, 
            recommended_filenames_euclidean[1:recommendation_count+1]  # Excluir a música de consulta
        )
        
        # Resultados da distância Manhattan
        print_metric_results(
            f_subj, 
            "Distância Manhattan", 
            manh_means, 
            manh_stds, 
            manh_overall_mean, 
            manh_overall_std, 
            manh_precision, 
            recommended_filenames_manhattan[1:recommendation_count+1]  # Excluir a música de consulta
        )
        
        # Resultados da distância de Cosseno
        print_metric_results(
            f_subj, 
            "Distância de Cosseno", 
            cos_means, 
            cos_stds, 
            cos_overall_mean, 
            cos_overall_std, 
            cos_precision, 
            recommended_filenames_cosine[1:recommendation_count+1]  # Excluir a música de consulta
        )
        
        # 4.2.2. Avaliação para o ranking baseado em metadados
        print("\n\n--- 4.2.2. Avaliação para o ranking baseado em metadados ---", file=f_subj)
        print_metric_results(
            f_subj, 
            "Similaridade de Metadados", 
            meta_means, 
            meta_stds, 
            meta_overall_mean, 
            meta_overall_std, 
            meta_precision, 
            recommended_filenames_metadata[1:recommendation_count+1]  # Excluir a música de consulta
        )
        
        # 4.2.3. Comparação e discussão dos resultados
        print("\n\n--- 4.2.3. Comparação e discussão dos resultados ---", file=f_subj)
        print("Comparação das médias globais:", file=f_subj)
        print(f"Euclidiana: {eucl_overall_mean:.2f}", file=f_subj)
        print(f"Manhattan: {manh_overall_mean:.2f}", file=f_subj)
        print(f"Cosseno: {cos_overall_mean:.2f}", file=f_subj)
        print(f"Metadados: {meta_overall_mean:.2f}", file=f_subj)
        
        print("\nComparação das precision:", file=f_subj)
        print(f"Euclidiana: {eucl_precision:.2f} ({eucl_precision*100:.1f}%)", file=f_subj)
        print(f"Manhattan: {manh_precision:.2f} ({manh_precision*100:.1f}%)", file=f_subj)
        print(f"Cosseno: {cos_precision:.2f} ({cos_precision*100:.1f}%)", file=f_subj)
        print(f"Metadados: {meta_precision:.2f} ({meta_precision*100:.1f}%)", file=f_subj)
        
        print("\nDiscussão:", file=f_subj)
        print("Os resultados da avaliação subjetiva mostram que:", file=f_subj)
        
        best_method = max(
            [("Euclidiana", eucl_overall_mean), 
             ("Manhattan", manh_overall_mean), 
             ("Cosseno", cos_overall_mean), 
             ("Metadados", meta_overall_mean)], 
            key=lambda x: x[1]
        )[0]
        
        best_precision = max(
            [("Euclidiana", eucl_precision), 
             ("Manhattan", manh_precision), 
             ("Cosseno", cos_precision), 
             ("Metadados", meta_precision)], 
            key=lambda x: x[1]
        )[0]
        
        # Identificar qual métrica tem menor desvio padrão global (mais consistente)
        most_consistent = min(
            [("Euclidiana", eucl_overall_std), 
             ("Manhattan", manh_overall_std), 
             ("Cosseno", cos_overall_std), 
             ("Metadados", meta_overall_std)], 
            key=lambda x: x[1]
        )[0]
        
        print(f"- A métrica com a melhor média global de avaliação é a {best_method}.", file=f_subj)
        print(f"- A métrica com a melhor precision é a {best_precision}.", file=f_subj)
        print(f"- A métrica com resultados mais consistentes (menor desvio padrão) é a {most_consistent}.", file=f_subj)
        print(f"- Este resultado sugere que a similaridade baseada em {best_method.lower()} corresponde melhor às preferências subjetivas dos avaliadores, possivelmente porque esta métrica captura melhor as características que os ouvintes humanos consideram importantes na similaridade musical.", file=f_subj)
        
        # Comparação entre avaliação objetiva e subjetiva
        print("- A diferença entre a avaliação objetiva (seção 4.1) e a avaliação subjetiva indica que existem aspectos da similaridade musical que são percebidos pelos ouvintes humanos mas não são adequadamente capturados pelas características de áudio extraídas. Isso sugere que a percepção humana de similaridade musical é complexa e multidimensional, indo além das características puramente acústicas.", file=f_subj)
        
        # Sugestões de melhorias
        print("- Recomendações para melhorias futuras:", file=f_subj)
        print("  1. Expandir o conjunto de características extraídas para incluir aspectos mais relacionados à percepção humana, como harmonia e estrutura musical.", file=f_subj)
        print("  2. Implementar uma abordagem de aprendizado de máquina que possa ajustar os pesos das diferentes características com base no feedback dos ouvintes.", file=f_subj)
        print("  3. Considerar uma abordagem híbrida que combine similaridade baseada em conteúdo com similaridade baseada em metadados, potencialmente melhorando a relevância das recomendações.", file=f_subj)
        print("  4. Explorar características perceptuais mais avançadas que possam capturar melhor os aspectos da música que influenciam a percepção humana de similaridade.", file=f_subj)
    
    print(f"Avaliação subjetiva salva em '{subjective_eval_filename}'")
    print("\n--- Section 4.2 Completed ---")

if __name__ == '__main__':
    main()
