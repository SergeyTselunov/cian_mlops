"""
Программа: Создание файла с датасетом.
Версия: 1.0
"""

from typing import Text

import pandas as pd


def create_data(data: pd.DataFrame, dataset_path: Text):
    """
    Создание дата сета из файла .csv
    :param data: датасет.
    :param dataset_path: Путь к месту хранения файла.
    """
    data.to_csv(dataset_path, index=False)
