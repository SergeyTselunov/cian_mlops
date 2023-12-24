"""
Программа: Добавление новых признаков
Версия: 1.0
"""
import json

import pandas as pd
from geopy import distance, Yandex
from geopy.extra.rate_limiter import RateLimiter


def metro_point(data: pd.DataFrame, metro_path, flg_geocode: bool, api: str):
    """
    Поиск координат метро через геокодирование либо через json файл.
    :param api: API для поиска координат.
    :param flg_geocode: Флаг геокодирования.
    :param metro_path: Путь к словарю наименований метро.
    :param data: Датасет.
    :return:
    """
    if flg_geocode:
        geolocator = Yandex(api_key=api)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

        metro = dict()
        for met in data["Город и метро"].unique():
            metro[met] = geocode(met)
        data["metro"] = data["Город и метро"].map(metro)
        data["координаты метро"] = data["metro"].apply(
            lambda x: tuple(x.point) if x else None
        )
    else:
        with open(metro_path) as file:
            metro = json.load(file)
        data["координаты метро"] = data["Город и метро"].map(metro)
    data["широта метро"] = data["координаты метро"].apply(lambda x: x[0])
    data["долгота метро"] = data["координаты метро"].apply(lambda x: x[1])


def address_point(data: pd.DataFrame, address_path, flg_geocode: bool, api: str):
    """
    Поиск координат дома через геокодирование либо через json файл.
    :param api: API для поиска координат.
    :param flg_geocode: Флаг геокодирования
    :param address_path: путь к словарю адресов.
    :param data: Датасет.
    :return:
    """
    if flg_geocode:
        geolocator = Yandex(api_key=api)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

        address = dict()
        for i in range(0, data["Адрес"].nunique()):
            adr = data["Адрес"].unique()[i]
            address[adr] = geocode(adr, timeout=1)
        data["address"] = data["Адрес"].map(address)
        data["координаты дома"] = data["address"].apply(
            lambda x: tuple(x.point) if x else None
        )

    else:
        with open(address_path) as file:
            address = json.load(file)
        data["координаты дома"] = data["Адрес"].map(address)
    data["широта дома"] = data["координаты дома"].apply(lambda x: x[0])
    data["долгота дома"] = data["координаты дома"].apply(lambda x: x[1])


def get_distance(data: pd.DataFrame, lon_center: float, lat_center: float):
    """
    Добавление дистанции до метро и до центра.
    :param lat_center: Широта центра Москвы
    :param lon_center: долгота центра Москвы
    :param data: датасет
    :return:
    """
    data["Расстояние до центра"] = data[["широта дома", "долгота дома"]].apply(
        lambda x: distance.distance((x[0], x[1]), (lat_center, lon_center)).km, axis=1
    )
    data["Расстояние до метро"] = data[
        ["широта дома", "долгота дома", "широта метро", "долгота метро"]
    ].apply(lambda x: distance.distance((x[0], x[1]), (x[2], x[3])).km, axis=1)


def get_district(data: pd.DataFrame):
    """
    Добавление признака Округ.
    :param data: Датасет
    """
    districts = [
        "ЦАО",
        "ЮАО",
        "ЮЗАО",
        "ЮВАО",
        "ЗАО",
        "СВАО",
        "ВАО",
        "САО",
        "СЗАО",
        "НАО (Новомосковский)",
        "ЗелАО",
    ]
    data["district"] = data["Адрес"].str.split(", ", expand=True)[1]

    district = dict()
    for i in range(0, len(data["Метро"])):
        if i not in district.keys():
            if data["district"][i] in districts:
                district[data["Метро"][i]] = data["district"][i]
    data["Округ"] = data["Метро"].map(district)

    for ind, dis in enumerate(data["Округ"]):
        if dis not in districts:
            data["Округ"][ind] = "Неизвестно"

    data["Округ"] = data["Округ"].str.split(" ", expand=True)[0]


def removing_excess(data: pd.DataFrame, flg_geocode: bool) -> pd.DataFrame:
    """
    Удаление лишних колонок.
    :param flg_geocode: Флаг геокодирования.
    :param data: Датасет
    :return: новый датасет
    """
    data_clean = data.drop(
        ["Город и метро", "координаты метро", "координаты дома", "district"], axis=1
    )
    if flg_geocode:
        data_clean = data_clean.drop(["metro", "address"], axis=1)
    return data_clean


def pipeline_geo(
    data: pd.DataFrame,
    flg_parse: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Пайплайн геокодирования.
    :param flg_parse: Включение геокодирования, в случае предсказания по введенным данным.
    :param data: Датасет
    :param kwargs:
    :return: новый датасет
    """
    data["Город и метро"] = "Москва, метро " + data["Метро"]
    metro_point(
        data=data,
        metro_path=kwargs["metro_path"],
        flg_geocode=flg_parse,
        api=kwargs["API_key"],
    )
    address_point(
        data=data,
        address_path=kwargs["address_path"],
        flg_geocode=flg_parse,
        api=kwargs["API_key"],
    )
    get_distance(
        data=data,
        lon_center=kwargs["lon_center"],
        lat_center=kwargs["lat_center"],
    )
    get_district(data)
    df = removing_excess(data, flg_geocode=flg_parse)
    return df
