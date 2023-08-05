import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mapping import MappingTrackId


def map_id_to_track(id, id_track_mapping):
    return id_track_mapping[id]


def create_json(id, id_track_mapping):
    track = MappingTrackId(id, map_id_to_track(id, id_track_mapping))
    track.save_track()


def load_data():
    df = pd.read_csv("data-train/arquivo_classificado.csv")
    id_track_mapping = dict(enumerate(df['id'].astype('category').cat.categories))
    df['id'] = df['id'].astype('category').cat.codes
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    return X, y, id_track_mapping


def train_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


def create_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)
    return model


def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss}")
    print(f"MAE: {loss[1]}")
    return loss


def save_model(model):
    model.save('trained-model/model.h5')


if __name__ == "__main__":
    X, y, id_track_mapping = load_data()

    for id in id_track_mapping:
        create_json(id, id_track_mapping)

    X_train, X_test, y_train, y_test, scaler = train_model(X, y)
    model = create_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


