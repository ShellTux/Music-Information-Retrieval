from common import SONGS

def main():
    statistics = SONGS.calculate_statistics()
    normalized_statistics = SONGS.normalize_features(statistics)
    print("Statistics Array:")
    print(statistics)
    print("Normalized Statistics Array:")
    print(normalized_statistics)

if __name__ == '__main__':
    main()
