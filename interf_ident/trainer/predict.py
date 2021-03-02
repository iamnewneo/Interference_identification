import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from interf_ident import config


def loss_fn(out, target):
    n_classes = len(config.CLASSES)
    return F.cross_entropy(out.view(-1, n_classes), target)


# def evaluate_model(model, data_loader):
#     all_predictions = torch.LongTensor([])
#     all_targets = torch.LongTensor([])
#     model = model.to(config.DEVICE)
#     result = {}
#     total = 0
#     correct = 0
#     total_loss = 0
#     with torch.no_grad():
#         for batch in data_loader:
#             X = batch["X"].to(config.DEVICE)
#             targets = batch["target"]
#             out = model(X)
#             out = out.to("cpu")
#             loss = loss_fn(out, targets)
#             out = F.softmax(out, dim=1)
#             _, predicted = torch.max(out.data, 1)

#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#             all_targets = torch.cat((all_targets, targets), axis=0)
#             all_predictions = torch.cat((all_predictions, predicted), axis=0)
#             total_loss += loss.item()

#     result["predictions"] = all_predictions
#     result["targets"] = all_targets
#     result["accuracy"] = (correct * 100) / total
#     result["loss"] = total_loss / len(data_loader)
#     return result


def evaluate_model(model, data_loader):
    model = model.to(config.DEVICE)
    result = {}
    pred_list = []
    soft_out_list = []
    target_list = []
    accuracy_metric = pl.metrics.Accuracy()
    with torch.no_grad():
        for batch in data_loader:
            X = batch["X"].to(config.DEVICE)
            targets = batch["target"]
            preds = model(X)
            soft_out = F.softmax(preds, dim=1)

            pred_list.append(preds)
            soft_out_list.append(soft_out)
            target_list.append(targets)

    result["predictions"] = torch.cat(pred_list, axis=0)
    _, result["prediction_labels"] = torch.max(result["predictions"], 1)
    result["prediction_proba"] = torch.cat(soft_out_list, axis=0)
    result["targets"] = torch.cat(target_list, axis=0)
    result["loss"] = loss_fn(result["predictions"], result["targets"]).item()
    result["accuracy"] = (
        100.0 * accuracy_metric(result["prediction_proba"], result["targets"]).item()
    )

    return result
