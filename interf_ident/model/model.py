import torch
from torch import nn
import torch.nn.functional as F
from interf_ident import config
import pytorch_lightning as pl


class InterIdentiModel(pl.LightningModule):
    def __init__(self):
        super(InterIdentiModel, self).__init__()
        n_classes = len(config.CLASSES)
        pool_size = (2, 2)
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxpool = nn.AdaptiveMaxPool2d(pool_size)
        self.fc1 = nn.Linear(128 * pool_size[0] * pool_size[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss_fn(self, out, target):
        return F.cross_entropy(out.view(-1, self.n_classes), target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        X = batch["X"]
        targets = batch["target"]
        out = self(X)
        loss = self.loss_fn(out, targets)
        out = F.softmax(out, dim=1)
        accuracy = self.train_accuracy(out, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch["X"]
        targets = batch["target"]
        out = self(X)
        loss = self.loss_fn(out, targets)
        out = F.softmax(out, dim=1)
        accuracy = self.valid_accuracy(out, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss, accuracy

    def training_epoch_end(self, train_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        print(f"Train Loss: {avg_val_loss:.2f}")

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            avg_val_loss = torch.tensor([x[0] for x in val_step_outputs]).mean()
            avg_val_acc = torch.tensor([x[1] for x in val_step_outputs]).mean()
            print(
                f"Epoch: {self.current_epoch} Val Acc: {avg_val_acc:.2f} Val Loss: {avg_val_loss:.2f} ",
                end="",
            )
