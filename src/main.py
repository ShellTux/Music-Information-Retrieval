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
    # Aplicar janela de Hanning
    if len(y) == 0:
        return 0.0
    window = np.hanning(len(y))
    y_windowed = y * window
    
    magnitude_spectrum: np.ndarray = np.abs(rfft(y_windowed))
    
    # As frequências devem corresponder ao n_fft usado, que é o comprimento da frame janelada
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length_for_fft)

    # Garantir que o array de frequências corresponde ao comprimento do magnitude_spectrum
    expected_bins = frame_length_for_fft // 2 + 1
    if len(magnitude_spectrum) != expected_bins:
        frequencies = frequencies[:len(magnitude_spectrum)]

    spectral_centroid_val = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) > 0 else 0.0
    return float(spectral_centroid_val)
 
def main():
    # --- SECÇÃO 2.1: Extração e Gravação de Features ---
    songs_bd_instance = None # Para armazenar a instância BD para uso posterior
    
    # Verificar se o ficheiro de features já existe
    if not os.path.exists(features_path):
        print(f"'{features_path}' não encontrado. A gerar features...")
        songs_bd_instance = BD('./MER_audio_taffc_dataset') 
        if not songs_bd_instance.songs:
            print("Nenhuma música foi processada com sucesso pela BD. A parar.")
            return
        songs_bd_instance.save_features_to_file(features_path)
        print(f"Features guardadas em '{features_path}'")
    else:
        print(f"A usar features existentes de '{features_path}'")
        # Carregar apenas metadados e lista de músicas sem processar ficheiros de áudio
        print("A carregar nomes de ficheiros de músicas a partir dos metadados sem processar ficheiros de áudio...")
        songs_bd_instance = BD('./MER_audio_taffc_dataset', load_features_only=True)
        if not songs_bd_instance.ordered_song_paths:
            print("Crítico: A instância BD não conseguiu determinar os caminhos ordenados das músicas. A parar.")
            return

    # Tentar carregar as features para verificar se a Secção 2.1 foi bem-sucedida ou se o ficheiro é utilizável
    try:
        # Esta variável irá conter min_vals, max_vals e normalized_feature_vectors
        all_feature_data = BD.load_features_from_file(features_path)
        if all_feature_data.shape[1] != 190: # 190 é o número esperado de features
             print(f"Aviso: Esperadas 190 features, mas encontradas {all_feature_data.shape[1]} em '{features_path}'. Os resultados podem estar incorretos.")
    except Exception as e:
        print(f"Erro ao carregar ou validar '{features_path}': {e}. Por favor, apague-o e execute novamente para regenerar.")
        return
        
    # --- SECÇÃO 2.2: Implementação e Avaliação do Spectral Centroid ---
    print("\n--- A iniciar Secção 2.2: Spectral Centroid ---")
    
    # Parâmetros para o spectral centroid
    sr_sc = 22050  # Taxa de amostragem consistente com o Librosa e os requisitos
    frame_length_sc = 2048 # Equivalente a 92.88ms @ 22050Hz
    hop_length_sc = 512    # Equivalente a 23.22ms @ 22050Hz

    # Ignorar avaliação do SC se o ficheiro de resultados já existir
    sc_results_path = 'resultados_metricas_sc.csv'
    if os.path.exists(sc_results_path):
        print(f"Ficheiro de resultados da avaliação SC '{sc_results_path}' já existe. A ignorar avaliação.")
    else:
        rmse_values_sc = []
        pearson_corr_values_sc = []
        
        # Usar um número limitado de músicas para testar o centroid para poupar tempo
        audio_files_for_sc_eval_path = './MER_audio_taffc_dataset'
        
        # Verificar se o diretório existe
        if not os.path.isdir(audio_files_for_sc_eval_path):
            print(f"Aviso: Diretório para ficheiros de áudio de avaliação SC não encontrado: {audio_files_for_sc_eval_path}")
            print("A ignorar Secção 2.2.2 (comparação SC com Librosa).")
            # Criar ficheiro de resultados fictício
            dummy_sc_results = np.zeros((900, 2)) # Assumindo 900 músicas
            np.savetxt(
                sc_results_path,
                dummy_sc_results,
                delimiter=',',
                header='RMSE_SC,Pearson_Correlation_SC',
                comments='',
                fmt=['%.6f', '%.6f']
            )
            print(f"Criado ficheiro fictício '{sc_results_path}'.")

        else:
            files_to_process_sc = [] # Inicializar
            if songs_bd_instance and hasattr(songs_bd_instance, 'ordered_song_paths') and songs_bd_instance.ordered_song_paths:
                files_to_process_sc = songs_bd_instance.ordered_song_paths
                print(f"A usar lista ordenada de {len(files_to_process_sc)} músicas da instância BD para avaliação SC.")
            else:
                print(f"Crítico: 'ordered_song_paths' da instância BD não está disponível ou está vazio.")
                print(f"Não é possível prosseguir com a avaliação SC ordenada. '{sc_results_path}' provavelmente terá todos os valores NaN.")
            
            if not files_to_process_sc:
                print(f"Nenhum ficheiro de áudio determinado para avaliação SC com base na instância BD.")
                # rmse_values_sc e pearson_corr_values_sc permanecerão vazios.
                # A lógica existente irá então guardar corretamente um CSV com NaNs.
            else:
                print(f"A processar SC para {len(files_to_process_sc)} ficheiros...")
                for audio_file_path in files_to_process_sc:
                    try:
                        y_sc, sr_loaded = librosa.load(audio_file_path, sr=sr_sc, mono=True)
                        if sr_loaded != sr_sc:
                             print(f"Aviso: Incompatibilidade de SR para {audio_file_path}. Esperado {sr_sc}, obtido {sr_loaded}.")

                        # Calcular spectral centroid com implementação manual
                        N_sc = len(y_sc)
                        pad_amount_sc = (hop_length_sc - ((N_sc - frame_length_sc) % hop_length_sc)) % hop_length_sc
                        y_sc_padded = np.concatenate([y_sc, np.zeros(pad_amount_sc)])
                        
                        num_manual_frames = (len(y_sc_padded) - frame_length_sc) // hop_length_sc + 1
                        # Garantir que num_manual_frames não é negativo se y_sc_padded for mais curto que frame_length_sc
                        if num_manual_frames < 0:
                            num_manual_frames = 0
                            
                        sc_manual_frames_list = [] # Usar uma lista para acrescentar e depois converter para array

                        if num_manual_frames > 0:
                            for i_frame in range(num_manual_frames):
                                start = i_frame * hop_length_sc
                                frame = y_sc_padded[start : start + frame_length_sc]
                                # compute_spectral_centroid espera uma frame não janelada, janelas internamente
                                sc_manual_frames_list.append(compute_spectral_centroid(frame, sr_sc, frame_length_sc))
                        
                        sc_manual = np.array(sc_manual_frames_list)

                        # Calcular spectral centroid usando librosa
                        sc_librosa_frames = librosa.feature.spectral_centroid(y=y_sc, sr=sr_sc, n_fft=frame_length_sc, hop_length=hop_length_sc)
                        sc_librosa = sc_librosa_frames.flatten() # Librosa retorna (1, T)

                        
                        sc_manual_aligned = sc_manual
                        if len(sc_manual_aligned) > 2:
                            sc_manual_aligned = sc_manual_aligned[:-2] # Remover os últimos 2 do manual
                        else:
                            sc_manual_aligned = np.array([]) # Fica vazio se não houver frames suficientes

                        sc_librosa_aligned = sc_librosa
                        if len(sc_librosa_aligned) > 2:
                            sc_librosa_aligned = sc_librosa_aligned[2:] # Remover os primeiros 2 do librosa
                        else:
                            sc_librosa_aligned = np.array([]) # Fica vazio se não houver frames suficientes
                        
                        # Garantir que sc_manual_aligned e sc_librosa_aligned têm o mesmo comprimento para comparação
                        min_len = min(len(sc_manual_aligned), len(sc_librosa_aligned))
                        
                        sc_manual_trimmed = np.array([]) # Inicializar como vazio
                        sc_librosa_trimmed = np.array([]) # Inicializar como vazio
                        
                        if min_len > 0:
                            sc_manual_trimmed = sc_manual_aligned[:min_len]
                            sc_librosa_trimmed = sc_librosa_aligned[:min_len]
                        
                        if min_len > 1: # Precisa de pelo menos 2 pontos para correlação de Pearson
                            rmse_sc = np.sqrt(np.mean((sc_manual_trimmed - sc_librosa_trimmed) ** 2))
                            pearson_corr_sc, _ = pearsonr(sc_manual_trimmed, sc_librosa_trimmed)
                            
                            # Lidar com NaNs de pearsonr se um vetor for constante
                            if np.isnan(pearson_corr_sc):
                                pearson_corr_sc = 0.0 # Ou 1.0 se for esperada correspondência perfeita para constante
                        elif min_len == 1 and sc_manual_trimmed[0] == sc_librosa_trimmed[0]:
                            rmse_sc = 0.0
                            pearson_corr_sc = 1.0 # Correspondência perfeita para ponto único
                        elif min_len > 0: # Ponto único, valores diferentes
                            rmse_sc = np.abs(sc_manual_trimmed[0] - sc_librosa_trimmed[0])
                            pearson_corr_sc = 0.0 # Nenhuma correlação definível, ou indefinida
                        else: # Nenhuma frame comparável
                            rmse_sc = np.nan 
                            pearson_corr_sc = np.nan

                        rmse_values_sc.append(rmse_sc)
                        pearson_corr_values_sc.append(pearson_corr_sc)

                    except Exception as e_sc:
                        print(f"Erro ao processar SC para {audio_file_path}: {e_sc}")
                        rmse_values_sc.append(np.nan)
                        pearson_corr_values_sc.append(np.nan)
                
                # Guardar resultados da avaliação SC
                num_results = len(rmse_values_sc)
                results_array_sc = np.full((900, 2), np.nan) # Inicializar com NaN
                
                # Preencher com resultados reais
                actual_results = np.column_stack((rmse_values_sc, pearson_corr_values_sc))
                results_array_sc[:num_results, :] = actual_results[:num_results, :] # Preencher o que temos
                
                np.savetxt(
                    sc_results_path, 
                    results_array_sc,
                    delimiter=',',
                    header='RMSE_SC,Pearson_Correlation_SC',
                    comments='',
                    fmt=['%.6f', '%.6f']
                )
                print(f"Resultados da avaliação do Spectral Centroid guardados em '{sc_results_path}'")


    # --- SECÇÃO 3: Implementação de Métricas de Similaridade ---
    print("\n--- A iniciar Secção 3: Métricas de Similaridade ---")
    
    # Carregar dados de features_statistics.csv 
    # all_feature_data já foi carregado no início de main()
    try:
        min_vals_dataset = all_feature_data[0, :]
        max_vals_dataset = all_feature_data[1, :]
        # As estatísticas já estão normalizadas, pois common.py foi modificado
        normalized_stats_all_songs = all_feature_data[2:, :] 
    except IndexError:
        print(f"ERRO: O ficheiro {features_path} não parece conter os dados esperados (min, max, stats).")
        print("Verifique se common.py está correto e regenere o ficheiro de features.")
        return
    except Exception as e_load_features:
         print(f"ERRO ao processar dados de {features_path}: {e_load_features}")
         return

    if normalized_stats_all_songs.shape[0] == 0:
        print(f"ERRO: Nenhuma estatística de música encontrada em {features_path}.")
        return
    if normalized_stats_all_songs.shape[1] != 190:
        print(f"AVISO: Esperadas 190 features normalizadas, mas encontradas {normalized_stats_all_songs.shape[1]}. As recomendações podem estar incorretas.")


    # Obter lista ordenada de nomes de ficheiros de músicas (para mapear índices para nomes)
    # Esta lógica agora usa o método da classe BD para garantir consistência com a ordem do ficheiro features_statistics.csv
    if songs_bd_instance is None:
        print("Erro: songs_bd_instance não inicializada. A reinicializar para obter a ordem das músicas.")
        songs_bd_instance = BD('./MER_audio_taffc_dataset', load_features_only=True)
        if not songs_bd_instance.ordered_song_paths:
             print("Crítico: A instância BD não conseguiu determinar os caminhos ordenados das músicas após a reinicialização. A parar.")
             return

    song_filenames_ordered = songs_bd_instance.get_ordered_song_filenames()

    if not song_filenames_ordered:
        print(f"ERRO: Nenhum nome de ficheiro MP3 obtido da instância BD. Verifique a inicialização da BD e o dataset.")
        return
    
    # Verificar consistência entre o número de músicas nas features e o número de nomes de músicas
    if len(song_filenames_ordered) != normalized_stats_all_songs.shape[0]:
        print(f"AVISO: O número de nomes de ficheiros ({len(song_filenames_ordered)}) não corresponde ao número de registos de estatísticas ({normalized_stats_all_songs.shape[0]}). As recomendações podem estar desalinhadas.")
        if not normalized_stats_all_songs.any(): 
             print("A abortar devido à falta de estatísticas.")
             return

    # 3.2.1. Para a query: Selecionar uma música de consulta
    # Encontrar o índice da música de consulta na nossa lista song_filenames_ordered
    query_song_target_name = "MT0000414517.mp3"
    query_song_index = -1
    try:
        query_song_index = song_filenames_ordered.index(query_song_target_name)
    except ValueError:
        print(f"ERRO: Música de consulta '{query_song_target_name}' não encontrada na lista ordenada de músicas.")
        print(f"Verifique o nome do ficheiro e confirme se existe no ficheiro de metadados e no dataset.")
        print(f"Primeiros ficheiros na lista ordenada: {song_filenames_ordered[:5]}")
        return

    if query_song_index >= normalized_stats_all_songs.shape[0]:
        print(f"ERRO: Índice da música de consulta ({query_song_index} para '{query_song_target_name}') fora dos limites ({normalized_stats_all_songs.shape[0]} músicas no dataset).")
        return
        
    query_song_normalized_stats = normalized_stats_all_songs[query_song_index]
    # query_song_filename agora refere-se apenas ao nome, ex: "MT0000414517.mp3"
    query_song_filename_for_print = song_filenames_ordered[query_song_index] 
    print(f"Música de Consulta: {query_song_filename_for_print}")

    # 3.1. Métricas de similaridade já estão disponíveis em scipy.spatial.distance

    # 3.2.2. Criar e guardar 3 matrizes de similaridade (vetores de distância)
    num_songs_in_db = normalized_stats_all_songs.shape[0]
    
    distances_euclidean = np.zeros(num_songs_in_db)
    distances_manhattan = np.zeros(num_songs_in_db)
    distances_cosine = np.zeros(num_songs_in_db)

    for i in range(num_songs_in_db):
        db_song_normalized_stats = normalized_stats_all_songs[i]
        distances_euclidean[i] = distance.euclidean(query_song_normalized_stats, db_song_normalized_stats)
        distances_manhattan[i] = distance.cityblock(query_song_normalized_stats, db_song_normalized_stats) # cityblock é Manhattan
        distances_cosine[i] = distance.cosine(query_song_normalized_stats, db_song_normalized_stats) # 1 - similaridade de cosseno

    # Guardar matrizes de similaridade (vetores de distância)
    np.savetxt('similarity_euclidean.csv', distances_euclidean.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_manhattan.csv', distances_manhattan.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    np.savetxt('similarity_cosine.csv', distances_cosine.reshape(-1, 1), delimiter=',', comments='', fmt='%.6f')
    print("Matrizes de similaridade (vetores de distância) guardadas em ficheiros .csv")

    # 3.3. Para a query, criar 3 rankings de similaridade (top 10)
    recommendation_count = 10

    def save_ranking_csv(filename, song_names_all_dataset, sorted_indices, distances, num_recommendations):
        # sorted_indices[0] é a própria query. Queremos as 10 *outras* músicas de topo.
        
        # Índices da query + top 10 recomendações
        ranked_indices_to_save = sorted_indices[:num_recommendations + 1] 
        
        with open(filename, 'w', encoding='utf-8') as f_rank:
            for i, idx in enumerate(ranked_indices_to_save):
                if idx < len(song_names_all_dataset):
                    song_name = song_names_all_dataset[idx] # Deve ser apenas o nome do ficheiro, ex: "MTxxxx.mp3"
                    distance_val = distances[idx]
                    f_rank.write(f"{song_name},{distance_val:.6f}\n")
                else:
                    print(f"Aviso: Índice {idx} fora dos limites para song_names_all_dataset ao guardar {filename}")

    # Ranking Euclidiano
    sorted_indices_euclidean = np.argsort(distances_euclidean)
    save_ranking_csv('ranking_euclidean.csv', song_filenames_ordered, sorted_indices_euclidean, distances_euclidean, recommendation_count)
    recommended_indices_euclidean = sorted_indices_euclidean[1 : recommendation_count + 1] # Top 10 excluindo a query

    # Ranking Manhattan
    sorted_indices_manhattan = np.argsort(distances_manhattan)
    save_ranking_csv('ranking_manhattan.csv', song_filenames_ordered, sorted_indices_manhattan, distances_manhattan, recommendation_count)
    recommended_indices_manhattan = sorted_indices_manhattan[1 : recommendation_count + 1] # Top 10 excluindo a query

    # Ranking Cosseno
    sorted_indices_cosine = np.argsort(distances_cosine)
    save_ranking_csv('ranking_cosine.csv', song_filenames_ordered, sorted_indices_cosine, distances_cosine, recommendation_count)
    recommended_indices_cosine = sorted_indices_cosine[1 : recommendation_count + 1] # Top 10 excluindo a query
    
    print("Rankings (top 10 + query) guardados em ficheiros ranking_*.csv")


    print(f"\nTop {recommendation_count} Recomendações para {query_song_filename_for_print} (excluindo a própria música):")
    
    # Para formatação semelhante ao ficheiro de validação
    # Guardar nomes e distâncias para impressão
    
    print("\nRecomendações - Distância Euclidiana:")
    recommended_filenames_euclidean = []
    recommended_distances_euclidean_values = []
    # Incluir a própria música de consulta para corresponder ao formato de saída do ficheiro de validação
    # A música de consulta é sorted_indices_euclidean[0]
    if sorted_indices_euclidean[0] < len(song_filenames_ordered):
        recommended_filenames_euclidean.append(song_filenames_ordered[sorted_indices_euclidean[0]])
        recommended_distances_euclidean_values.append(distances_euclidean[sorted_indices_euclidean[0]])
        
    for idx in recommended_indices_euclidean: # Top 10, excluindo a query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_euclidean)}. {song_filenames_ordered[idx]} (Distância: {distances_euclidean[idx]:.4f})")
            recommended_filenames_euclidean.append(song_filenames_ordered[idx])
            recommended_distances_euclidean_values.append(distances_euclidean[idx])
        else:
            print(f"{len(recommended_filenames_euclidean)}. Índice {idx} fora dos limites para nomes de ficheiros.")

    print("\nRecomendações - Distância Manhattan:")
    recommended_filenames_manhattan = []
    recommended_distances_manhattan_values = []
    if sorted_indices_manhattan[0] < len(song_filenames_ordered):
        recommended_filenames_manhattan.append(song_filenames_ordered[sorted_indices_manhattan[0]])
        recommended_distances_manhattan_values.append(distances_manhattan[sorted_indices_manhattan[0]])

    for idx in recommended_indices_manhattan: # Top 10, excluindo a query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_manhattan)}. {song_filenames_ordered[idx]} (Distância: {distances_manhattan[idx]:.4f})")
            recommended_filenames_manhattan.append(song_filenames_ordered[idx])
            recommended_distances_manhattan_values.append(distances_manhattan[idx])
        else:
            print(f"{len(recommended_filenames_manhattan)}. Índice {idx} fora dos limites para nomes de ficheiros.")
            
    print("\nRecomendações - Distância Cosseno:")
    recommended_filenames_cosine = []
    recommended_distances_cosine_values = []
    if sorted_indices_cosine[0] < len(song_filenames_ordered):
        recommended_filenames_cosine.append(song_filenames_ordered[sorted_indices_cosine[0]])
        recommended_distances_cosine_values.append(distances_cosine[sorted_indices_cosine[0]])

    for idx in recommended_indices_cosine: # Top 10, excluindo a query
        if idx < len(song_filenames_ordered):
            print(f"{len(recommended_filenames_cosine)}. {song_filenames_ordered[idx]} (Distância: {distances_cosine[idx]:.4f})")
            recommended_filenames_cosine.append(song_filenames_ordered[idx])
            recommended_distances_cosine_values.append(distances_cosine[idx])
        else:
            print(f"{len(recommended_filenames_cosine)}. Índice {idx} fora dos limites para nomes de ficheiros.")
            
    print("\n--- Secção 3 Concluída ---")

    # --- SECÇÃO 4.1: Avaliação Objetiva ---
    print("\n--- A iniciar Secção 4.1: Avaliação Objetiva ---")

    metadata_file_path = './MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    try:
        metadata_df = pd.read_csv(metadata_file_path)
    except FileNotFoundError:
        print(f"ERRO: Ficheiro de metadados '{metadata_file_path}' não encontrado.")
        return

    # Garantir que a coluna 'Song' existe (nome do ficheiro sem .mp3)
    if 'Song' not in metadata_df.columns:
        print(f"ERRO: Coluna 'Song' não encontrada em '{metadata_file_path}'.")
        return
    
    # Indexar DataFrame por ID da música para consulta fácil
    metadata_df = metadata_df.set_index('Song')

    # 4.1.1 Obter metadados para a música de consulta
    query_song_id_stem = Path(query_song_filename_for_print).stem
    if query_song_id_stem not in metadata_df.index:
        print(f"ERRO: Metadados para a música de consulta '{query_song_filename_for_print}' (ID: {query_song_id_stem}) não encontrados no ficheiro CSV.")
        return
    query_metadata = metadata_df.loc[query_song_id_stem]

    def get_metadata_similarity_score(query_meta, target_meta):
        score = 0
        
        # Comparar Artista
        if pd.notna(query_meta['Artist']) and pd.notna(target_meta['Artist']) and query_meta['Artist'] == target_meta['Artist']:
            score += 1
            
        # Comparar Géneros e Estados de Espírito (MoodsStrSplit e GenresStr)
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
            metadata_similarity_scores[i] = 0 # Ou outra estratégia

    # Guardar matriz de similaridade baseada no contexto
    np.savetxt('similarity_metadata.csv', metadata_similarity_scores.reshape(-1, 1), delimiter=',', comments='', fmt='%d')
    print("Matriz de similaridade baseada em metadados guardada em 'similarity_metadata.csv'")

    # Obter ranking de metadados (top 10 + query, ou top 11 para exibição)
    # Scores são "quanto maior, melhor"
    sorted_indices_metadata = np.argsort(metadata_similarity_scores)[::-1] 
    
    # Para exibição
    num_recommendations_display_metadata = recommendation_count + 1 
    recommended_indices_metadata_display = sorted_indices_metadata[:num_recommendations_display_metadata]
    
    recommended_filenames_metadata = [song_filenames_ordered[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]
    recommended_scores_metadata = [metadata_similarity_scores[idx] for idx in recommended_indices_metadata_display if idx < len(song_filenames_ordered)]

    # Para cálculo de precisão, considerar as 10 *outras* músicas de topo
    # (excluindo a própria música de consulta que estaria em sorted_indices_metadata[0])
    # Se a música de consulta não for a primeira, esta lógica precisa de ajuste, mas espera-se que seja.
    if sorted_indices_metadata[0] == query_song_index:
         relevant_indices_for_precision_metadata = sorted_indices_metadata[1 : recommendation_count + 1]
    else:
        # Se a query não for a mais similar a si mesma (improvável com esta métrica, mas por segurança)
        # Remover a query e pegar as 10 restantes de topo
        temp_sorted_metadata = [idx for idx in sorted_indices_metadata if idx != query_song_index]
        relevant_indices_for_precision_metadata = temp_sorted_metadata[:recommendation_count]

    relevant_set_for_precision = set(relevant_indices_for_precision_metadata)

    # 4.1.2 Calcular métrica de precisão
    # recommended_indices_euclidean, etc., já são o top 10 excluindo a query
    
    precision_euclidean = len(set(recommended_indices_euclidean).intersection(relevant_set_for_precision)) / recommendation_count
    precision_manhattan = len(set(recommended_indices_manhattan).intersection(relevant_set_for_precision)) / recommendation_count
    precision_cosine = len(set(recommended_indices_cosine).intersection(relevant_set_for_precision)) / recommendation_count

    # Apresentar resultados
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
        # Converter scores para int para impressão, como no ficheiro de exemplo
        print(f"{np.array(recommended_scores_metadata, dtype=int)}", file=f_out)

        print(f"\nPrecision de:  {precision_euclidean * 100:.1f}", file=f_out)
        print(f"Precision dm:  {precision_manhattan * 100:.1f}", file=f_out)
        print(f"Precision dc:  {precision_cosine * 100:.1f}", file=f_out)

    print(f"\nRelatório de ranking e precisão guardado em: {output_txt_filename}")
    print("\n--- Secção 4.1 Concluída ---")

    # --- SECÇÃO 4.2: Avaliação Subjetiva ---
    print("\n--- A iniciar Secção 4.2: Avaliação Subjetiva ---")
    
    # Definindo constantes para a avaliação subjetiva
    RELEVANCE_THRESHOLD = 2.5  # Score mínimo para considerar uma recomendação relevante
    
    # Carregar dados reais de avaliação a partir de ficheiros
    ratings_dir = './subjective_ratings'
    
    # Inicializar listas para armazenar as avaliações de cada membro
    euclidean_ratings_list = []
    manhattan_ratings_list = []
    cosine_ratings_list = []
    metadata_ratings_list = []
    
    NUM_GROUP_MEMBERS = 0 # Será atualizado com base nos ficheiros carregados

    # Inicializar arrays numpy vazios. Serão preenchidos se os dados forem carregados.
    euclidean_subjective_ratings = np.array([])
    manhattan_subjective_ratings = np.array([])
    cosine_subjective_ratings = np.array([])
    metadata_subjective_ratings = np.array([])

    # Verificar se existem ficheiros de avaliação
    if os.path.exists(ratings_dir) and any(f.endswith('.csv') for f in os.listdir(ratings_dir)):
        print(f"A carregar avaliações subjetivas do diretório '{ratings_dir}'...")
        
        # Carregar cada ficheiro de avaliações
        for ratings_file in sorted([f for f in os.listdir(ratings_dir) if f.endswith('.csv')]):
            file_path = os.path.join(ratings_dir, ratings_file)
            try:
                ratings_df = pd.read_csv(file_path)
                
                # Extrair avaliações para cada métrica, se existirem no ficheiro
                if 'euclidean' in ratings_df.columns:
                    member_euclidean = ratings_df['euclidean'].values
                    if len(member_euclidean) == recommendation_count:
                        euclidean_ratings_list.append(member_euclidean)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_euclidean)} avaliações para euclidiana, esperava {recommendation_count}. Ficheiro ignorado para esta métrica.")
                
                if 'manhattan' in ratings_df.columns:
                    member_manhattan = ratings_df['manhattan'].values
                    if len(member_manhattan) == recommendation_count:
                        manhattan_ratings_list.append(member_manhattan)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_manhattan)} avaliações para manhattan, esperava {recommendation_count}. Ficheiro ignorado para esta métrica.")
                
                if 'cosine' in ratings_df.columns:
                    member_cosine = ratings_df['cosine'].values
                    if len(member_cosine) == recommendation_count:
                        cosine_ratings_list.append(member_cosine)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_cosine)} avaliações para cosseno, esperava {recommendation_count}. Ficheiro ignorado para esta métrica.")
                
                if 'metadata' in ratings_df.columns:
                    member_metadata = ratings_df['metadata'].values
                    if len(member_metadata) == recommendation_count:
                        metadata_ratings_list.append(member_metadata)
                    else:
                        print(f"Aviso: '{ratings_file}' contém {len(member_metadata)} avaliações para metadados, esperava {recommendation_count}. Ficheiro ignorado para esta métrica.")
                
                # Considera-se que um ficheiro contribui se pelo menos a euclidiana foi lida com sucesso
                # No entanto, NUM_GROUP_MEMBERS será baseado na consistência mais abaixo

            except Exception as e:
                print(f"Erro ao carregar {ratings_file}: {e}")
        
        # Determinar NUM_GROUP_MEMBERS com base na consistência das listas carregadas
        # Usamos a euclidiana como referência, mas verificamos as outras.
        if euclidean_ratings_list:
            potential_num_members = len(euclidean_ratings_list)
            # Verificar se todas as listas povoadas têm o mesmo tamanho que a euclidiana
            all_lists = {'manhattan': manhattan_ratings_list, 'cosine': cosine_ratings_list, 'metadata': metadata_ratings_list}
            consistent = True
            for metric_name, metric_list in all_lists.items():
                if metric_list and len(metric_list) != potential_num_members:
                    print(f"Aviso: Inconsistência no número de avaliações para {metric_name} ({len(metric_list)}) em comparação com euclidiana ({potential_num_members}).")
                    consistent = False # Ou decidir por uma estratégia de preenchimento/ignorar membro
            
            NUM_GROUP_MEMBERS = potential_num_members # Usar o número de membros que forneceram euclidiana
                                                    # Avisos sobre inconsistências já foram dados.
            print(f"Número de membros com avaliações (baseado em euclidiana): {NUM_GROUP_MEMBERS}")

            if NUM_GROUP_MEMBERS > 0:
                euclidean_subjective_ratings = np.array(euclidean_ratings_list)
                
                # Para as outras métricas, apenas converte se o tamanho for consistente com NUM_GROUP_MEMBERS
                if len(manhattan_ratings_list) == NUM_GROUP_MEMBERS:
                    manhattan_subjective_ratings = np.array(manhattan_ratings_list)
                else:
                    print(f"Aviso: Avaliações de Manhattan incompletas ({len(manhattan_ratings_list)}/{NUM_GROUP_MEMBERS}). Preenchendo com zeros.")
                    manhattan_subjective_ratings = np.zeros((NUM_GROUP_MEMBERS, recommendation_count))
                
                if len(cosine_ratings_list) == NUM_GROUP_MEMBERS:
                    cosine_subjective_ratings = np.array(cosine_ratings_list)
                else:
                    print(f"Aviso: Avaliações de Cosseno incompletas ({len(cosine_ratings_list)}/{NUM_GROUP_MEMBERS}). Preenchendo com zeros.")
                    cosine_subjective_ratings = np.zeros((NUM_GROUP_MEMBERS, recommendation_count))
                
                if len(metadata_ratings_list) == NUM_GROUP_MEMBERS:
                    metadata_subjective_ratings = np.array(metadata_ratings_list)
                else:
                    print(f"Aviso: Avaliações de Metadados incompletas ({len(metadata_ratings_list)}/{NUM_GROUP_MEMBERS}). Preenchendo com zeros.")
                    metadata_subjective_ratings = np.zeros((NUM_GROUP_MEMBERS, recommendation_count))
            # Se NUM_GROUP_MEMBERS for 0 aqui (nenhuma euclidiana válida), cairá no else externo.
    else:
        print(f"Aviso: Diretório de avaliações '{ratings_dir}' não encontrado ou não contém ficheiros .csv válidos.")
        NUM_GROUP_MEMBERS = 0 # Garante que é zero se nenhum ficheiro foi carregado

    # Proceder com cálculos e escrita de ficheiro apenas se houver dados de membros
    if NUM_GROUP_MEMBERS > 0:
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
        
        # Guardar resultados em um ficheiro
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
            
            print("\n\n--- Resultados ---", file=f_subj)
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
            
        print(f"Avaliação subjetiva salva em '{subjective_eval_filename}'")
    else: # NUM_GROUP_MEMBERS == 0
        print("ERRO: Nenhuma avaliação subjetiva válida foi carregada. Os resultados da avaliação subjetiva não puderam ser calculados.")
        subjective_eval_filename = "subjective_evaluation_results.txt"
        with open(subjective_eval_filename, 'w', encoding='utf-8') as f_subj:
            print("=== AVALIAÇÃO SUBJETIVA ===", file=f_subj)
            print("\nERRO: Nenhuma avaliação subjetiva válida foi carregada dos ficheiros .csv.", file=f_subj)
            print("Os resultados da avaliação subjetiva não puderam ser calculados.", file=f_subj)
        print(f"Relatório de erro da avaliação subjetiva salvo em '{subjective_eval_filename}'")

    print("\n--- Secção 4.2 Concluída ---")

if __name__ == '__main__':
    main()
