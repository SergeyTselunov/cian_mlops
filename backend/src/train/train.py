"""
Программа: Тренировка данных
Версия: 1.0
"""
import json

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna import Study
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective(
    trial,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_folds: int = 5,
    random_state: int = 26,
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    lgb_params = {
        "metric": trial.suggest_categorical("metric", ["rmse"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 100]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
    }

    cv_folds = KFold(n_splits=n_folds, shuffle=True)
    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_folds.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]
        y_test_exp = np.exp(y_test)

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
        model = LGBMRegressor(**lgb_params, silent=True)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test_exp)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=-1,
        )
        preds = model.predict(x_test)
        preds_exp = np.exp(preds)
        cv_predicts[idx] = mean_absolute_error(y_test_exp, preds_exp)
    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMRegressor tuning, Study]
    """

    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )
    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)

    # save params
    best_params = kwargs["params_path"]
    with open(best_params, "w") as f:
        json.dump(study.best_params, f)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    data_train_: pd.DataFrame,
    data_val: pd.DataFrame,
    target: str,
    metric_path: str,
    params_path: str,
    top_params: bool,
    study: Study = None,
) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param top_params: Флаг при обучении на последних параметрах.
    :param params_path: Путь к параметрам.
    :param data_train: Тренировочный датасет
    :param data_test: тестовый датасет
    :param data_train_: тренировочный набор для валидационных данных
    :param data_val: тестовый набор для валидационных данных
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )
    x_train_, x_val, y_train_, y_val = get_train_test_data(
        data_train=data_train_, data_test=data_val, target=target
    )
    y_test_exp = np.exp(y_test)
    y_val_exp = np.exp(y_val)
    eval_set = [(x_val, y_val_exp)]

    if top_params:
        with open(params_path) as file:
            top = json.load(file)
        clf = LGBMRegressor(**top, silent=True, verbose=-1)
    else:
        clf = LGBMRegressor(**study.best_params, silent=True, verbose=-1)
    clf.fit(
        x_train_,
        y_train_,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=False,
        early_stopping_rounds=100,
    )
    pred = clf.predict(x_test)
    pred_exp = np.exp(pred)
    # save metrics
    save_metrics(
        data_x=x_test, data_y=y_test_exp, y_pred=pred_exp, metric_path=metric_path
    )

    return clf
