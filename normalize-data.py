import pandas as pd
import numpy as np


def load_dataset():
    return pd.read_csv("data-train/taylor_swift_spotify.csv")


def filter_dataset(dataset):
    dataset_filtered = dataset[~dataset["album"].isin([
        "Fearless",
        "Speak Now",
        "Red",
        "reputation Stadium Tour Surprise Song Playlist",
        "1989",
        "evermore",
        "Red (Deluxe Edition)",
        "Speak Now World Tour Live",
        "Fearless Platinum Edition",
        "Live From Clear Channel Stripped 2008",
        "folklore",
        "Midnights (3am Edition)",
        "Midnights"
    ])]

    return dataset_filtered


def classify_music(row):
    if row['valence'] < 0.5:
        return 0.0
    else:
        return 1.0


def classify_energy(row):
    if row['energy'] < 0.5:
        return 0.0
    else:
        return 1.0


def classify_danceability(row):
    if row['danceability'] < 0.5:
        return 0.0
    else:
        return 1.0


def classify_acousticness(row):
    if row['acousticness'] < 0.5:
        return 0.0
    else:
        return 1.0


def main():
    df = load_dataset()
    df = filter_dataset(df)

    df['mood'] = df.apply(classify_music, axis=1)
    df['vibe'] = df.apply(classify_energy, axis=1)
    df['dance_type'] = df.apply(classify_danceability, axis=1)
    df['acoustic_type'] = df.apply(classify_acousticness, axis=1)

    # Salvar somente as colunas que serão utilizadas
    df = df[[
        'id',
        'mood',
        'vibe',
        'dance_type',
        'acoustic_type',
    ]]

    # Converter as colunas para o tipo float32
    float32_columns = ['mood', 'vibe', 'dance_type', 'acoustic_type']
    df[float32_columns] = df[float32_columns].astype(np.float32)

    # Salvando o resultado em um novo arquivo CSV
    output_file = "data/arquivo_classificado.csv"
    df.to_csv(output_file, index=False, header=True)


if __name__ == '__main__':
    main()
