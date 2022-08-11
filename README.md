# Optuna meets Weights and Biases

This repo provides the examples codes used in the medium post titled [Optuna meets Weights and Biases](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893).
---

## Installation

## `pip`

```bash
pip install wandb optuna scikit-learn torch torchvision plotly
```
## `conda`

```bash
conda env create -f environment.yml
conda activate optuna-wandb
```

---

### Updated 🚀 [11-Aug-2022]: Add `as_multirun=True` example to make [`part-1`](./part-1/wandb_optuna.py) simpler

In forthcoming optuna v3, optuna's wandb callback provides `as_multirun` option to trace an objective function optimised by iterative way, e.g., stochastic gradient descent. Thanks to this feature, we can combine optuna and wandb more easily.
