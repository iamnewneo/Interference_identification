import torch
from torch import nn
import torch.nn.functional as F
from interf_ident import config
import pytorch_lightning as pl


class InterIdentiModel(pl.LightningModule):
    def __init__(self):
        super(InterIdentiModel, self).__init__()
        n_classes = len(config.CLASSES)
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.n_classes), target)

    def configure_optimizers(self):
        LR = config.LR
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        X = batch["X"]
        targets = batch["target"]
        out = self(X)
        loss = self.loss_fn(out, targets)
        accuracy = self.train_accuracy(out, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch["X"]
        targets = batch["target"]
        out = self(X)
        loss = self.loss_fn(out, targets)
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