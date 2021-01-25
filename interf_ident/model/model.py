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
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.inp1 = nn.Linear(2, 128)
        self.inp2 = nn.Linear(128, 256)
        self.inp3 = nn.Linear(256, 512)
        self.inp4 = nn.Linear(512, 128)
        self.inp5 = nn.Linear(128, 64)
        self.inp6 = nn.Linear(64, n_classes)
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = F.relu(self.inp1(x))
        x = F.relu(self.inp2(x))
        x = F.relu(self.inp3(x))
        x = F.relu(self.inp4(x))
        x = F.relu(self.inp5(x))
        x = F.relu(self.inp6(x))
        return F.log_softmax(x)

    def loss_fn(self, out, target):
        return F.nll_loss(out.view(-1, self.n_classes), target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LR)
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