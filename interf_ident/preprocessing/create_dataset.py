import os
import pickle
import numpy as np
from interf_ident import config
from sklearn.model_selection import train_test_split


def create_dataset(start_db=None, valid_ratio=0.2):
    file_list = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    data_base_path = f"{config.BASE_PATH}/data"
    if all([os.path.isfile(f"{data_base_path}/{x}") for x in file_list]):
        print("Data File Already Present")
        return

    db_list = list(range(-20, 21, 2))
    start_index_of_data = 0
    if start_db is not None:
        start_index_of_data = db_list.index(start_db)
        print(f"Data Created for SNR starting at: {start_db}")
    data_path = f"{config.BASE_PATH}/data/data_iq.p"
    label_path = f"{config.BASE_PATH}/data/labels.p"

    # Shape: (15, 21, 715, 128, 2)
    with open(data_path, "rb") as f:
        x_data = pickle.load(f, encoding="latin1")
        x_data = x_data[:, start_index_of_data:, :, :, :]

    # Shape: (15, 21, 715)
    with open(label_path, "rb") as f:
        y_data = pickle.load(f, encoding="latin1")
        y_data = y_data[:, start_index_of_data:, :]

    x_data = x_data.reshape((-1, 128, 2))

    y_data = y_data.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, stratify=y_data, test_size=valid_ratio, random_state=42
    )

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
