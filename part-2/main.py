"""
Mainly copied from https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_simple.py
"""

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb


def objective(trial):

    data = fetch_olivetti_faces()
    x_train, x_valid, y_train, y_valid = train_test_split(data["data"], data["target"])

    params = {
        "n_estimators": trial.suggest_int("min_samples_leaf", 1, 256, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 256, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 256, log=True),
    }

    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)
    score = accuracy_score(y_valid, pred)

    return score


samplers = (
    optuna.samplers.RandomSampler,
    optuna.samplers.TPESampler,
)

num_runs = 5
n_trials = 30

for sampler in samplers:
    for _ in range(num_runs):
        wandb_kwargs = {
            "project": "sklearn-wandb",
            "entity": "nzw0301",
            "config": {"sampler": sampler.__name__},
            "reinit": True,
        }

        wandbc = WeightsAndBiasesCallback(
            metric_name="val_accuracy", wandb_kwargs=wandb_kwargs
        )

        study = optuna.create_study(direction="maximize", sampler=sampler())
        study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])

        f = "best_{}".format
        for param_name, param_value in study.best_trial.params.items():
            wandb.run.summary[f(param_name)] = param_value

        wandb.run.summary["best accuracy"] = study.best_trial.value

        wandb.log(
            {
                "optuna_optimization_history": optuna.visualization.plot_optimization_history(
                    study
                ),
                "optuna_param_importances": optuna.visualization.plot_param_importances(
                    study
                ),
            }
        )

        wandb.finish()
