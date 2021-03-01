import numpy as np
import torchvision.transforms as transforms
from interf_ident import config
from interf_ident.preprocessing.create_dataset import create_dataset
from interf_ident.trainer.trainer import model_trainer
from interf_ident.data_loader.data_loader import create_data_loader
from interf_ident.trainer.predict import evaluate_model
from interf_ident.utils.util import get_confusion_matrix


def main():

    start_db = -10
    create_dataset(start_db, valid_ratio=0.2)
    print("Loading Data")
    X_train = np.load(f"{config.BASE_PATH}/data/X_train.npy")
    X_test = np.load(f"{config.BASE_PATH}/data/X_test.npy")
    y_train = np.load(f"{config.BASE_PATH}/data/y_train.npy")
    y_test = np.load(f"{config.BASE_PATH}/data/y_test.npy")
    if config.ENV == "dev":
        X_train = X_train[:100]
        X_test = X_test[:100]
        y_train = y_train[:100]
        y_test = y_test[:100]

    print(f"Train Size: {X_train.shape[0]}. Shape: {X_train.shape}")
    preprocessing = None
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # )
    transform = None

    train_loader = create_data_loader(
        X_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        preprocessing=preprocessing,
        transform=transform,
    )
    val_loader = create_data_loader(
        X_test,
        y_test,
        batch_size=config.BATCH_SIZE,
        preprocessing=preprocessing,
        transform=transform,
    )

    trainer = model_trainer(train_loader, val_loader, progress_bar_refresh_rate=0)
    model = trainer.get_model()
    result = evaluate_model(model, data_loader=val_loader)
    print(f"\nTest Loss: {result['loss']:.2f}")
    print(f"Test Accuracy: {result['accuracy']:.2f}")
    confusion_matrix = get_confusion_matrix(result["targets"], result["predictions"])
    print(confusion_matrix)


if __name__ == "__main__":
    main()
