"""
Программа: Предобработка данных
Версия: 1.0
"""
import json
import warnings

import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


def replace_comma(
    data: pd.DataFrame, lst: list, inplace_true: bool = False
) -> pd.DataFrame:
    """
    Замена запятых на точки в столбцах и удаление лишних символов (\xa0)
    :param data: дата фрейм
    :param lst: список из названий колонок
    :param inplace_true: True - изменения сразу применятся к data, False - создается новый дата фрейм
    """
    data_new = data.copy()
    data_new[lst] = (
        data[lst]
        .apply(lambda x: x.str.split("\xa0").str[0].str.replace(",", ".").astype(float))
        .copy()
    )
    if inplace_true is True:
        data = data_new.copy()
        return data
    else:
        return data_new


def sum_obj(obj: pd.Series) -> pd.Series:
    """
    Удаление лишней информации из значений признака и преобразование в числовой тип.
    :param obj: Колонка из датасета.
    """

    temp_obj_1 = pd.to_numeric(obj.str.split(" ", expand=True)[0], errors="coerce")
    temp_obj_2 = pd.to_numeric(obj.str.split(" ", expand=True)[2], errors="coerce")
    temp = pd.concat([temp_obj_1, temp_obj_2], axis=1)
    temp = temp.sum(axis=1, min_count=1).astype("Int64")
    return temp


def ar_part(data: pd.DataFrame, col: str, main_col: str) -> pd.Series:
    """
    Заполнение пропусков в зависимости от отношения площади признака к общей площади.
    :param data: Датасет
    :param col: Название колонки площади необходимого признака.
    :param main_col: Название колонки общей площади.
    """
    part = round((data[col] / data[main_col]).median(), 2)
    col_new = data[col].fillna(data[main_col] * part)
    return col_new


def replace_group(
    data: pd.DataFrame, main_col: str, col_1: str, col_2: str
) -> pd.DataFrame:
    """
    Заполнение модой через groupby.
    :param data: Дата фрейм.
    :param main_col: Название колонки по которой идет группировка.
    :param col_1: Название колонки в которой происходит заполнение пустых значений.
    :param col_2: Название колонки в которой происходит заполнение пустых значений.
    :return: Датасет.
    """
    for col in data[[col_1, col_2]]:
        data[col] = data[col].fillna(
            data.groupby(main_col)[col].transform(lambda x: x.mode()[0])
        )
        data[col] = data[col].astype(int)
    data_new = data.copy()
    return data_new


def replace_elev(row: pd.Series) -> int:
    """
    Заполнение пропусков в признаке кол-во лифтов в зависимости от этажности здания согласно СНиП 31-01-2003
    :param row: строка дата фрейма.
    """

    if row["Количество лифтов"] == 0:
        floors = int(row["Этажность здания"])
        if floors < 6:
            return 0
        elif floors < 10:
            return 1
        elif floors < 20:
            return 2
        elif floors < 25:
            return 3
        else:
            return 4
    else:
        return row["Количество лифтов"]


def drop_rooms(data: pd.DataFrame, col: str) -> pd.Series:
    """
    Удаление лишних уникальных значений в признаке 'Кол-во комнат', квартиры с большим кол-вом комнат и апартаменты.
    :param data: Датасет
    :param col: Название колонки "кол-во комнат"
    """
    to_drop = (
        data[col]
        .loc[
            lambda x: (x == "Многокомнатная")
            | (x == "Многокомнатные")
            | (x == "Апартаменты")
        ]
        .index
    )
    data_new = data.drop(index=to_drop).copy()
    data_new[col][data_new[col] == "Студия,"] = 1
    data_new[col] = pd.to_numeric(data_new[col], errors="coerce")
    return data_new


def anomaly_drop(data: pd.DataFrame) -> pd.DataFrame:
    """Удаление аномалий в данных: Отрицательные значения и значения меньше 2 в "Высота потолков"
    :param data: Датасет
    """
    df_num = data.select_dtypes(include=np.number)
    drop_neg = []
    for i, j in df_num.iterrows():
        for k in j:
            if k < 0:
                drop_neg.append(i)
    drop_neg += data[data["Высота потолков"] < 2].index.tolist()
    df_fin = data.drop(index=drop_neg)
    df_fin = df_fin.reset_index(drop=True)
    return df_fin


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списка с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(columns=drop_columns, axis=1)
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def pipeline_preprocess(
    data: pd.DataFrame, flg_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных.
    :param flg_evaluate: Флаг для evaluate.
    :param data: Датасет.
    :return: Датасет.
    """
    drop_list = [
        "Газоснабжение",
        "Год сдачи",
        "Дом",
        "Строительная серия",
        "Аварийность",
        "Подъезды",
    ]
    data = data.drop(columns=drop_list).copy()
    # Обработка данных.
    # Проверка идет ли предсказание.
    if not flg_evaluate:
        data = data.drop(data["Цена"].isna().loc[lambda x: x == True].index)

    data_clean = data.drop(
        data["Метро"].isna().loc[lambda x: x == True].index
    ).reset_index(drop=True)

    data_clean = replace_comma(
        data_clean,
        ["Жилая площадь", "Площадь кухни", "Общая площадь", "Высота потолков"],
    ).copy()

    data_clean["Время до метро"] = (
        data_clean["Время до метро"].str.split(" ").str[0].copy()
    )
    data_clean["Время до метро"] = pd.to_numeric(
        data_clean["Время до метро"], errors="coerce"
    )
    data_clean["Время до метро"] = data_clean["Время до метро"].fillna(
        data_clean["Время до метро"].median()
    )
    data_clean["Время до метро"] = data_clean["Время до метро"].copy().astype(int)

    data_clean[["Санузел", "Количество лифтов", "Балкон/лоджия"]] = data_clean[
        ["Санузел", "Количество лифтов", "Балкон/лоджия"]
    ].apply(sum_obj)

    floor = data_clean["Этаж"].str.split(" из ", expand=True)
    data_clean["Этаж"] = floor[0]
    data_clean["Этажность здания"] = floor[1]
    data_clean["Этаж"] = pd.to_numeric(data_clean["Этаж"], errors="coerce")

    data_clean["Кол-во комнат"] = (
        data_clean["Название"]
        .str.split(" ", expand=True)[0]
        .str.split("-", expand=True)[0]
    )

    data_clean = drop_rooms(data_clean, "Кол-во комнат")
    data_clean = data_clean.drop(columns=["Название"])

    data_clean["Тип жилья"] = data_clean["Тип жилья"].str.split(" ").str[0]

    # Заполнение пропусков.
    for i in data_clean[["Жилая площадь", "Площадь кухни"]]:
        data_clean[i] = ar_part(data_clean, i, "Общая площадь")

    curr_year = datetime.now().year
    data_clean["Год постройки"] = (
        data_clean["Год постройки"].fillna(curr_year + 1).astype(int)
    )

    data_clean = replace_group(
        data=data_clean,
        main_col="Кол-во комнат",
        col_1="Санузел",
        col_2="Балкон/лоджия",
    ).copy()

    data_clean["Количество лифтов"].fillna(0, inplace=True)
    data_clean["Количество лифтов"] = data_clean["Количество лифтов"].astype(int)
    data_clean["Этажность здания"] = data_clean["Этажность здания"].astype(int)
    data_clean["Количество лифтов"] = data_clean.apply(
        lambda row: replace_elev(row), axis=1
    )

    data_clean["Высота потолков"] = data_clean["Высота потолков"].fillna(
        data_clean["Высота потолков"].median()
    )

    data_clean["Отделка"] = data_clean["Отделка"].fillna("Неизвестно")

    to_fill = data_clean[
        [
            "Отопление",
            "Вид из окон",
            "Ремонт",
            "Тип дома",
            "Тип перекрытий",
            "Парковка",
            "Мусоропровод",
        ]
    ]
    for column in to_fill:
        data_clean[column] = data_clean[column].fillna(data_clean[column].mode()[0])

    data_fin = anomaly_drop(data_clean)

    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        data_fin = check_columns_evaluate(
            data=data_fin, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=data_fin,
            drop_columns=kwargs["drop_columns_unique"],
            unique_values_path=kwargs["unique_values_path"],
        )
    return data_fin
