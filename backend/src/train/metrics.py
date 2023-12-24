"""
Программа: Получение метрик
Версия: 1.0
"""

import json
import numpy as np
import yaml
from sklearn.metrics import (
    r2_score,
    mean_squared_log_error,
    mean_absolute_error,
    mean_squared_error,
)
import pandas as pd


def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray, x_test: np.ndarray) -> float:
    """Коэффициент детерминации (множественная регрессия)"""
    n_objects = len(y_true)
    n_features = x_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def mpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean percentage error"""
    return np.mean((y_true - y_pred) / y_true) * 100


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error"""
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percent Error"""
    return np.sum(np.abs(y_pred - y_true)) / np.sum(y_true) * 100


def rmsle(y_true: np.ndarray, y_pred: np.ndarray):
    """
    The Root Mean Squared Log Error (RMSLE) metric
    Логарифмическая ошибка средней квадратичной ошибки
    """
    try:
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    except:
        return None


def create_dict_metrics(
    y_test: pd.Series, y_pred: pd.Series, x_test: pd.DataFrame
) -> dict:
    """
    Получение словаря с метриками для задачи регрессии и запись в словарь
    :param x_test: тестовые данные
    :param y_test: реальные данные
    :param y_pred: предсказанные значения
    :return: словарь с метриками
    """
    dict_metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 3),
        "MSE": round(mean_squared_error(y_test, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        "RMSLE": round(rmsle(y_test, y_pred), 3),
        "R2_adjusted": round(r2_adjusted(y_test, y_pred, x_test), 3),
        "MPE_%": round(mpe(y_test, y_pred), 3),
        "MAPE_%": round(mape(y_test, y_pred), 3),
        "WAPE_%": round(wape(y_test, y_pred), 3),
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, y_pred: pd.Series, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param y_pred: предсказанные значения
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=data_y,
        y_pred=y_pred,
        x_test=data_x,
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str, cross_val: bool = False) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if not cross_val:
        with open(config["train"]["metrics_path"]) as json_file:
            metrics = json.load(json_file)
    else:
        with open(config["train"]["metrics_cross_path"]) as json_file:
            metrics = json.load(json_file)

    return metrics
