import pickle
import numpy as np
from interf_ident import config
from sklearn.model_selection import train_test_split


def create_dataset():
    data_path = f"{config.BASE_PATH}/data/data_iq.p"
    label_path = f"{config.BASE_PATH}/data/labels.p"

    with open(data_path, "rb") as f:
        x_data = pickle.load(f, encoding="latin1")

    with open(label_path, "rb") as f:
        y_data = pickle.load(f, encoding="latin1")

    x_data = x_data.reshape((-1, 2))

    y_data = y_data.reshape((-1, 1))
    y_data = np.repeat(y_data, repeats=128, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, stratify=y_data, test_size=0.25, random_state=42
    )

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    
    del x_data
    del y_data

    with open(f"{config.BASE_PATH}/data/X_train.npy", "wb") as f:
        np.save(f, X_train)
    with open(f"{config.BASE_PATH}/data/X_test.npy", "wb") as f:
        np.save(f, X_test)
    with open(f"{config.BASE_PATH}/data/y_train.npy", "wb") as f:
        np.save(f, y_train)
    with open(f"{config.BASE_PATH}/data/y_test.npy", "wb") as f:
        np.save(f, y_test)


if __name__ == "__main__":
    create_dataset()
