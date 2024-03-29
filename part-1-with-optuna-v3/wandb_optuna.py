"""
The main code is based on https://github.com/optuna/optuna-examples/blob/63fe36db4701d5b230ade04eb2283371fb2265bf/pytorch/pytorch_simple.py
"""

import wandb
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from optuna.integration.wandb import WeightsAndBiasesCallback


# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 100
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
STUDY_NAME = "pytorch-optimization"


def train(optimizer, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Limiting training data for faster epochs.
        if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
            break

        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def validate(model, valid_loader):
    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Limiting validation data.
            if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = model(data)
            # Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

    return accuracy


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


# Get the data loaders of FashionMNIST dataset.
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        DIR, train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=BATCHSIZE,
    shuffle=True,
)
valid_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
    batch_size=BATCHSIZE,
    shuffle=True,
)

# Initialise wandb callback
wandb_kwargs = {
    "project": "optuna-v3-wandb-pytorch",
    "entity": "nzw0301",
    "reinit": True,
}

wandbc = WeightsAndBiasesCallback(
    metric_name="final validation accuracy", wandb_kwargs=wandb_kwargs, as_multirun=True
)


# Add decorator to track objective
@wandbc.track_in_wandb()
def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in range(EPOCHS):

        train(optimizer, model, train_loader)
        val_accuracy = validate(model, valid_loader)
        trial.report(val_accuracy, epoch)

        # report validation accuracy to wandb
        wandb.log(data={"validation accuracy": val_accuracy}, step=epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            raise optuna.exceptions.TrialPruned()

    # report the final validation accuracy to wandb
    wandb.run.summary["final accuracy"] = val_accuracy
    wandb.run.summary["state"] = "complated"

    return val_accuracy


study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=100, timeout=600, callbacks=[wandbc])
