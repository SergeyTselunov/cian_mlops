"""
Программа: Преобразование признаков.
Версия: 1.0
"""

import numpy as np
import pandas as pd
from scipy.stats import stats
import re
from transliterate import translit


def outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Поиск и удаление выбросов по нижней границе.
    :param data: Датасет
    :return: Новый датасет
    """
    targ = data["Цена за квадрат лог"].copy()
    q1 = targ.quantile(q=0.25)
    iqr = stats.iqr(targ)
    lower_bound = q1 - (1.5 * iqr)
    df_clean = data[data["Цена за квадрат лог"] > lower_bound]
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def translate(data: pd.DataFrame):
    """
    Перевод названий столбцов на английский язык.
    :param data: Датасет
    :return: новый датасет
    """
    cols_translit = [
        translit(x, language_code="ru", reversed=True).replace(" ", "_")
        for x in data.columns
    ]
    data.columns = cols_translit
    data_new = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    return data_new


def drop_features(data: pd.DataFrame, drop_cols: list):
    """
    Удаление признаков и преобразование типа object в category
    :param drop_cols: Список с названиями колонок
    :param data: датасет
    :return: новый датасет
    """

    data_clean = data.drop(columns=drop_cols)
    cat_cols = data_clean.select_dtypes("object").columns
    data_clean[cat_cols] = data_clean[cat_cols].astype("category")
    return data_clean


def pipeline_features(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs):
    """
    Пайплайн преобразования признаков.
    :param flg_evaluate:
    :param data: Датасет
    :param kwargs:
    :return:
    """
    if not flg_evaluate:
        # Логарифмирование целевого признака
        data["Цена за квадрат лог"] = np.log(data["Цена за квадрат"])

        # Работа с выбросами
        data = outliers(data)

        data = drop_features(data=data, drop_cols=kwargs["drop_columns_unique"])

    # Перевод русских названий
    data = translate(data)
    # Удаление колонок для обучения
    data = drop_features(data=data, drop_cols=kwargs["drop_columns"])

    return data
