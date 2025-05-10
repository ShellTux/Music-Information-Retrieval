from pathlib import Path
import random
from scipy import stats
from sys import stderr
import librosa
import numpy as np
import os
import pandas as pd

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
        assert self.filename.exists(), f"Ficheiro não encontrado: {self.filename}"

        self.extract_features()

    def extract_features(self) -> Features:
        y, sr = librosa.load(self.filename, sr=22050, mono=True)
        features = Features()

        # Extrair features espectrais
        features.mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features.spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Extrair features temporais
        features.f0 = librosa.yin(y, fmin=20, fmax=sr/2)
        # Limpar valores de F0
        features.f0[features.f0 == sr/2] = 0
        
        features.rms = librosa.feature.rms(y=y)
        features.zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        
        # Extrair tempo
        features.tempo = librosa.feature.tempo(y=y, sr=sr)

        # Obter variável de tempo
        features.time = np.arange(len(y)) / sr

        self.features = features

        return features

class BD:
    songs: list[Song] = []
    ordered_song_paths: list[Path] = []

    def __init__(self, directory: str | Path, metadata_filename: str = "panda_dataset_taffc_metadata.csv", load_features_only: bool = False):
        self.songs = []
        self.ordered_song_paths = []
        
        base_dir = Path(directory)
        metadata_path = base_dir / metadata_filename

        # 1. Procurar todos os MP3s e mapear o ID da música (stem) para o caminho completo
        all_mp3_files_map: dict[str, Path] = {}
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.mp3'):
                    p = Path(os.path.join(root, file))
                    all_mp3_files_map[p.stem] = p
        
        if not all_mp3_files_map:
            print(f"Nenhum ficheiro .mp3 encontrado em {base_dir} e nos seus subdiretórios.", file=stderr)
            return

        # 2. Ler CSV de metadados
        try:
            metadata_df = pd.read_csv(metadata_path)
        except FileNotFoundError:
            print(f"Ficheiro de metadados não encontrado: {metadata_path}", file=stderr)
            print("A recorrer à ordem de varrimento do sistema de ficheiros (diretório_pai, nome_ficheiro).", file=stderr)
            # Recorrer ao comportamento antigo se os metadados estiverem em falta
            fallback_paths: list[Path] = []
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.mp3'):
                        fallback_paths.append(Path(os.path.join(root, file)))
            fallback_paths.sort(key=lambda path: (path.parent.name, path.name))
            self.ordered_song_paths = fallback_paths
        else:
            # 3. Construir lista ordenada de músicas com base no CSV
            if 'Song' not in metadata_df.columns:
                print(f"Crítico: Coluna 'Song' não encontrada em {metadata_path}.", file=stderr)
                print("A recorrer à ordem de varrimento do sistema de ficheiros (diretório_pai, nome_ficheiro).", file=stderr)
                # Recorrer ao comportamento antigo
                fallback_paths: list[Path] = []
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith('.mp3'):
                            fallback_paths.append(Path(os.path.join(root, file)))
                fallback_paths.sort(key=lambda path: (path.parent.name, path.name))
                self.ordered_song_paths = fallback_paths
            else:
                processed_paths_in_csv_order: list[Path] = []
                print(f"A ler a ordem das músicas do ficheiro de metadados: {metadata_path}")
                for song_id_from_csv in metadata_df['Song']:
                    song_id_stem = str(song_id_from_csv).strip()
                    if song_id_stem in all_mp3_files_map:
                        processed_paths_in_csv_order.append(all_mp3_files_map[song_id_stem])
                    else:
                        print(f"Aviso: ID da música '{song_id_stem}' dos metadados não encontrado nos ficheiros MP3. A ignorar.", file=stderr)
                self.ordered_song_paths = processed_paths_in_csv_order
        
        if not self.ordered_song_paths:
             print("Nenhum caminho de música determinado (a partir do CSV ou do fallback). Não é possível prosseguir.", file=stderr)
             return

        # 4. Processar músicas na ordem determinada (apenas se não estiver no modo load_features_only)
        if not load_features_only:
            total_songs_to_process = len(self.ordered_song_paths)
            for i, path in enumerate(self.ordered_song_paths):
                try:
                    self.songs.append(Song(path))
                except AssertionError as e:
                    print(f"A ignorar {path} devido a erro: {e}", file=stderr)
                    continue # Ignorar esta música e continuar com a próxima
                except Exception as e_song:
                    print(f"Erro ao processar a música {path}: {e_song}. A ignorar.", file=stderr)
                    continue

                index = i + 1
                ratio = index / total_songs_to_process * 100 if total_songs_to_process > 0 else 0
                print(f'\r{index:3} / {total_songs_to_process} ({ratio:.2f}%) concluído. {path}', end='')
            
            # Verificar se alguma música foi processada com sucesso
            if not self.songs:
                print("\n\nAviso: Nenhuma música foi processada com sucesso e adicionada à BD.", file=stderr)
            elif len(self.songs) < total_songs_to_process:
                print(f"\n\nAviso: Processadas {len(self.songs)} músicas, mas esperadas {total_songs_to_process} com base nos caminhos. Algumas músicas podem ter sido ignoradas devido a erros.", file=stderr)

            print() # Nova linha após a barra de progresso
    
    def get_ordered_song_filenames(self) -> list[str]:
        """Retorna uma lista de nomes de ficheiros de músicas (nome.ext) na ordem em que foram processados."""
        return [p.name for p in self.ordered_song_paths if p is not None] # Garantir que o caminho não é None

    def _get_7_stats(self, data_array: np.ndarray) -> np.ndarray:
        """Função auxiliar para calcular 7 estatísticas para um dado array de dados 1D."""
        data_array = np.asarray(data_array).flatten() 
        if data_array.size == 0:
            # Retornar zeros se os dados da feature estiverem vazios para manter a forma do array
            return np.zeros(7)
        
        # Substituir NaNs e Infs que podem vir de features do librosa ou do processamento de f0
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.mean(data_array)
        std = np.std(data_array)
        # Verificar se o array é constante antes de calcular skew/kurtosis para evitar avisos/NaNs
        if np.all(data_array == data_array[0] if data_array.size > 0 else True):
            skewness = 0.0
            kurt = -3.0 # Curtose de uma constante é tipicamente -3 (distribuição normal = 0)
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

            # MFCC (13 bandas) -> 13 * 7 = 91 features
            for band_idx in range(f.mfcc.shape[0]):
                current_song_stats.extend(self._get_7_stats(f.mfcc[band_idx, :]))
            
            # Spectral Centroid -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_centroid))
            
            # Spectral Bandwidth -> 7 features
            current_song_stats.extend(self._get_7_stats(f.spectral_bandwidth))
            
            # Spectral Contrast (librosa default é 7 linhas) -> 7 * 7 = 49 features
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
            
            # Tempo -> 1 feature (apenas o valor)
            # Garantir que o tempo é um escalar; librosa.feature.tempo retorna um array (geralmente com um elemento)
            tempo_value = f.tempo[0] if isinstance(f.tempo, np.ndarray) and f.tempo.size > 0 else (f.tempo if np.isscalar(f.tempo) else 0)
            current_song_stats.append(tempo_value)
            
            all_song_features_stats.append(current_song_stats)
            
        return np.array(all_song_features_stats)

    def normalize_features(self, statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_vals = np.min(statistics, axis=0)
        max_vals = np.max(statistics, axis=0)
        
        # Evitar divisão por zero: se max == min, a feature é constante, o valor normalizado deve ser 0
        range_vals = max_vals - min_vals
        # Criar uma versão segura de range_vals para divisão
        # Onde range_vals é 0, usar 1 para evitar divisão por zero (o numerador também será 0)
        safe_range_vals = np.where(range_vals == 0, 1, range_vals)
        
        normalized_statistics = (statistics - min_vals) / safe_range_vals
        
        # Garantir que quaisquer NaNs são convertidos para números (ex: 0)
        normalized_statistics = np.nan_to_num(normalized_statistics, nan=0.0)

        return normalized_statistics, min_vals, max_vals

    def save_features_to_file(self, filename: str | Path):
        unnormalized_statistics = self.calculate_statistics()
        
        if unnormalized_statistics.size == 0:
            print("Nenhuma estatística foi calculada. Não é possível guardar no ficheiro.", file=stderr)
            return

        normalized_stats, min_vals, max_vals = self.normalize_features(unnormalized_statistics)
        
        # Guardar min_vals, max_vals e depois as estatísticas normalizadas
        # Isto permite que a função load_features_from_file reconstrua/desnormalize se necessário,
        # ou simplesmente use as normalizadas.
        # A primeira linha são os mínimos, a segunda são os máximos.
        # As linhas subsequentes são os vetores de features normalizados para cada música.
        data_to_save = np.vstack((min_vals, max_vals, normalized_stats))
        
        # Especificar o formato para garantir consistência e legibilidade
        # Usar um formato que preserve precisão suficiente para as features
        np.savetxt(filename, data_to_save, delimiter=',', fmt='%.8e') 

    @staticmethod
    def load_features_from_file(filename: str | Path) -> np.ndarray:
        """ 
        Carrega features de um ficheiro.
        Assume que a primeira linha são os valores mínimos, a segunda os valores máximos,
        e as restantes linhas são os vetores de features normalizados.
        Retorna o array completo conforme guardado (min, max, dados_normalizados).
        """
        return np.loadtxt(filename, delimiter=',')

    def print(self):
        print(f'Found {len(self.songs)} songs in the database:')
        for song in self.songs:
            print(f'  {song.filename}')
