"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""
import os

import joblib
import numpy as np
import pandas as pd
import yaml


from ..data.get_data import get_dataset
from ..transform.feature_engineering import pipeline_features
from ..transform.geo_code import pipeline_geo
from ..transform.transform import pipeline_preprocess


def pipeline_evaluate(
    config_path,
    dataset: pd.DataFrame = None,
    data_path: str = None,
    flg_input: bool = False,
):
    """
    Предобработка входных данных и получение предсказаний
    :param flg_input: Флаг для Evaluate_input.
    :param dataset: Датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """
    # get params
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:  # Проверка существует ли датасет.
        dataset = get_dataset(dataset_path=data_path)

    if not flg_input:  # Проверка: Вводятся ли данные в ручную.
        dataset = pipeline_preprocess(data=dataset, **train_config)

    dataset = pipeline_geo(data=dataset, flg_parse=flg_input, **preprocessing_config)
    dataset = pipeline_features(data=dataset, flg_evaluate=True, **train_config)
    # Загрузка модели.
    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = np.exp(model.predict(dataset)).tolist()

    return prediction
