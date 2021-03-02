import torch
from torch import nn
import torch.nn.functional as F
from interf_ident import config
import pytorch_lightning as pl


class InterIdentiModel(pl.LightningModule):
    def __init__(self):
        super(InterIdentiModel, self).__init__()
        n_classes = len(config.CLASSES)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Linear(128 * 2 * 2, 64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, n_classes),
        # )
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # self.maxpool = nn.AdaptiveMaxPool2d(pool_size)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 16 * 2, 128)
        self.fc2 = nn.Linear(128, n_classes)
        # self.fc3 = nn.Linear(32, n_classes)
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = self.maxpool(F.relu(self.conv1(x)))
        # print(f"conv1: {x.shape}")
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # print(f"conv2: {x.shape}")
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        # print(f"conv3: {x.shape}")
        # x = self.maxpool(F.relu(self.conv4(x)))
        # print(f"conv4: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"FC Ip: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # print(f"FC1: {x.shape}")
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # print(f"FC2: {x.shape}")
        # x = self.fc3(x)
        # print(f"FC3: {x.shape}")
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
        # soft_out = F.softmax(out, dim=1)
        # accuracy = self.train_accuracy(soft_out, targets)
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch["X"]
        targets = batch["target"]
        out = self(X)
        loss = self.loss_fn(out, targets)
        # soft_out = F.softmax(out, dim=1)
        # accuracy = self.valid_accuracy(soft_out, targets)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", accuracy, prog_bar=True)
        return loss, out, targets

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        self.temp_train_loss = avg_train_loss

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.running_sanity_check:
            # avg_val_loss = torch.tensor([x[0] for x in val_step_outputs]).mean()
            # avg_val_acc = torch.tensor([x[1] for x in val_step_outputs]).mean()
            preds = torch.cat([x[1] for x in val_step_outputs], axis=0)
            targets = torch.cat([x[2] for x in val_step_outputs], axis=0)
            actual_loss = self.loss_fn(preds, targets)
            soft_out = F.softmax(preds, dim=1)
            actual_acc = self.valid_accuracy(soft_out, targets)
            print(
                f"Epoch: {self.current_epoch} Val Acc: {actual_acc:.2f}"
                f" Val Loss: {actual_loss:.2f} Train Loss: {self.temp_train_loss:.2f}"
            )
