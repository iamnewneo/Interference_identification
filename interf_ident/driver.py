import numpy as np
from interf_ident import config
from interf_ident.preprocessing.create_dataset import create_dataset
from interf_ident.trainer.trainer import model_trainer
from interf_ident.data_loader.data_loader import create_data_loader

# from interf_ident.trainer.predict import evaluate_model


def main():

    # create_dataset()
    print("Loading Data")
    X_train = np.load(f"{config.BASE_PATH}/data/X_train.npy")[:100]
    X_test = np.load(f"{config.BASE_PATH}/data/X_test.npy")[:100]
    y_train = np.load(f"{config.BASE_PATH}/data/y_train.npy")[:100]
    y_test = np.load(f"{config.BASE_PATH}/data/y_test.npy")[:100]

    train_loader = create_data_loader(X_train, y_train, batch_size=config.BATCH_SIZE)
    val_loader = create_data_loader(X_test, y_test, batch_size=config.BATCH_SIZE)
    print("Starting Training")
    trainer = model_trainer(train_loader, val_loader, progress_bar_refresh_rate=0)
    model = trainer.get_model()
    # result = evaluate_model(model, data_loader=val_loader)
    # print(f"Test Loss: {result['loss']:.2f}")
    # print(f"Test Accuracy: {result['accuracy']:.2f}")


if __name__ == "__main__":
    main()