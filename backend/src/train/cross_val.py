"""
Программа: Проверка модели на кросс валидации
Версия: 1.0
"""
import json

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def cross_validation(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model,
    params: dict,
    eval_metric: str = None,
    early_stop: bool = False,
    early_stopping_rounds: int = 100,
    num_folds: int = 5,
    random_state: int = 26,
    shuffle: bool = True,
):
    """
    Кросс-валидация для регрессии.
    :param x_train: X_train
    :param y_train: y_train
    :param x_test: x_test
    :param model: Модель для обучения
    :param params: параметры для обучения
    :param eval_metric: Метрика для обучения
    :param early_stop: Флаг ранней остановки
    :param early_stopping_rounds: Ранняя остановка
    :param num_folds: Кол-во фолдов
    :param random_state: Рандом стейт.
    :param shuffle: Флаг для перемешивания.
    :return: score_oof, predictions_test
    """
    folds = KFold(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
    score_oof = []
    predictions_test = []

    for fold, (train_index, test_index) in enumerate(folds.split(x_train, y_train)):
        X_train_, X_val = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        y_val_exp = np.exp(y_val)
        lgb = model(**params)

        if early_stop:
            if eval_metric is None:
                lgb.fit(
                    X_train_,
                    y_train_,
                    eval_set=[(X_val, y_val_exp)],
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                lgb.fit(
                    X_train_,
                    y_train_,
                    eval_set=[(X_val, y_val_exp)],
                    eval_metric=eval_metric,
                    verbose=-1,
                    early_stopping_rounds=early_stopping_rounds,
                )
        else:
            lgb.fit(X_train_, y_train_)

        y_pred_val = lgb.predict(X_val)
        y_pred_val_exp = np.exp(y_pred_val)
        y_pred = lgb.predict(x_test)
        y_pred_exp = np.exp(y_pred)

        print(
            "Fold:",
            fold + 1,
            "MAE SCORE %.3f" % mean_absolute_error(y_val_exp, y_pred_val_exp),
        )
        print("---")

        # oof list
        score_oof.append(mean_absolute_error(y_val_exp, y_pred_val_exp))
        # holdout list
        predictions_test.append(y_pred_exp)

    return score_oof, predictions_test


def cross_train(data_train, data_test, target, n_folds, params, metric_path):
    """
    Обучение на кросс-валидации
    :param data_train: датасет с тренировочными данными
    :param data_test: датасет с тестовыми данными
    :param target: целевая переменная
    :param n_folds: кол-во фолдов
    :param params: параметры для обучения
    :param metric_path: пусть к параметрам.
    :return: Значение переобучения
    """
    # Преобразование категориальнных данных после загрузки из json.
    cat_cols = data_train.select_dtypes("object").columns
    data_train[cat_cols] = data_train[cat_cols].astype("category")

    cat_cols = data_test.select_dtypes("object").columns
    data_test[cat_cols] = data_test[cat_cols].astype("category")

    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )
    y_test_exp = np.exp(y_test)

    score_oof, predictions_test = cross_validation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        num_folds=n_folds,
        early_stop=True,
        eval_metric="rmse",
        early_stopping_rounds=100,
        model=LGBMRegressor,
        params=params,
    )

    test_pred = np.mean(predictions_test, axis=0)
    oof = np.mean(score_oof)
    hold = mean_absolute_error(y_test_exp, test_pred)

    print("MAE mean OOF: %.3f, std: %.3f" % (oof, np.std(score_oof)))
    print("MAE HOLDOUT: %.3f" % hold)

    save_metrics(
        data_x=x_test, data_y=y_test_exp, y_pred=test_pred, metric_path=metric_path
    )
    overfit = round(np.abs((oof - hold) / hold) * 100, 2)
    return overfit


def pipeline_cross(config_path):
    """
    Пайплайн кросс-валдиации.
    :param config_path: Конфигурационный файл
    :return: Значение переобучения
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    df_train = pd.read_csv(preprocessing_config["train_path_proc"])
    df_test = pd.read_csv(preprocessing_config["test_path_proc"])

    with open(train_config["params_path"]) as json_file:
        params = json.load(json_file)

    overfit = cross_train(
        data_train=df_train,
        data_test=df_test,
        target=train_config["target_column"],
        n_folds=train_config["n_folds"],
        params=params,
        metric_path=train_config["metrics_cross_path"],
    )
    return overfit
