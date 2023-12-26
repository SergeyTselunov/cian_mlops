"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""
import json
from io import BytesIO

import pandas as pd
import streamlit as st
from datetime import datetime
import requests


def evaluate_input(unique_data_path: str, endpoint: object):
    """
    Получение входных данных путем ввода в UI и получение предсказания.
    :param endpoint: endpoint
    :param unique_data_path: пусть к уникальным значениям
    """

    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для ввода данных, используем уникальные значения
    st.sidebar.write("#")
    Address = st.sidebar.text_input("Адрес")
    st.sidebar.markdown("Пример: Москва, ЦАО, р-н Тверской, Тверская ул., 12С7")
    Metro = st.sidebar.text_input("Метро")
    Vremja_do_metro = st.sidebar.slider(
        "Время до метро",
        min_value=min(unique_df["Время до метро"]),
        max_value=max(unique_df["Время до метро"]),
        step=1,
    )
    Obschaja_ploschad = st.sidebar.number_input(
        "Общая площадь",
        min_value=min(unique_df["Общая площадь"]),
        max_value=max(unique_df["Общая площадь"]),
        step=0.1,
    )
    Zhilaja_ploschad = st.sidebar.number_input(
        "Жилая площадь",
        min_value=min(unique_df["Жилая площадь"]),
        max_value=max(unique_df["Жилая площадь"]),
        step=0.1,
    )
    Ploschad_kuhni = st.sidebar.number_input(
        "Площадь кухни",
        min_value=min(unique_df["Площадь кухни"]),
        max_value=max(unique_df["Площадь кухни"]),
        step=0.1,
    )
    Etazh = st.sidebar.slider("Этаж", min_value=1, max_value=max(unique_df["Этаж"]))
    God_postrojki = st.sidebar.number_input(
        "Год постройки",
        min_value=min(unique_df["Год постройки"]),
        max_value=(datetime.now().year + 2),
    )
    Tip_zhilja = st.sidebar.selectbox("Тип жилья", unique_df["Тип жилья"])
    Vysota_potolkov = st.sidebar.slider(
        "Высота потолков",
        min_value=min(unique_df["Высота потолков"]),
        max_value=max(unique_df["Высота потолков"]),
        step=0.1,
    )
    Sanuzel = st.sidebar.slider(
        "Кол-во санузлов",
        min_value=min(unique_df["Санузел"]),
        max_value=max(unique_df["Санузел"]),
    )
    Vid_iz_okon = st.sidebar.selectbox("Вид из окон", unique_df["Вид из окон"])
    Remont = st.sidebar.selectbox("Ремонт", unique_df["Ремонт"])
    Musoroprovod = st.sidebar.selectbox("Мусоропровод", unique_df["Мусоропровод"])
    Kolichestvo_liftov = st.sidebar.slider(
        "Кол-во лифтов",
        min_value=min(unique_df["Количество лифтов"]),
        max_value=max(unique_df["Количество лифтов"]),
    )
    Tip_doma = st.sidebar.selectbox("Тип дома", unique_df["Тип дома"])
    Tip_perekrytij = st.sidebar.selectbox("Тип перекрытий", unique_df["Тип перекрытий"])
    Parkovka = st.sidebar.selectbox("Парковка", unique_df["Парковка"])
    Otoplenie = st.sidebar.selectbox("Отопление", unique_df["Отопление"])
    Balkonlodzhija = st.sidebar.slider(
        "Кол-во балконов\лоджий",
        min_value=min(unique_df["Балкон/лоджия"]),
        max_value=max(unique_df["Балкон/лоджия"]),
    )
    Otdelka = st.sidebar.selectbox("Отделка", unique_df["Отделка"])
    Etazhnost_zdanija = st.sidebar.slider(
        "Этажность здания",
        min_value=min(unique_df["Этажность здания"]),
        max_value=max(unique_df["Этажность здания"]),
    )
    Kolvo_komnat = st.sidebar.slider(
        "Кол-во комнат",
        min_value=min(unique_df["Кол-во комнат"]),
        max_value=max(unique_df["Кол-во комнат"]),
    )

    dict_data = {
        "Address": Address,
        "Metro": Metro,
        "Vremja_do_metro": Vremja_do_metro,
        "Obschaja_ploschad": Obschaja_ploschad,
        "Zhilaja_ploschad": Zhilaja_ploschad,
        "Ploschad_kuhni": Ploschad_kuhni,
        "Etazh": Etazh,
        "God_postrojki": God_postrojki,
        "Tip_zhilja": Tip_zhilja,
        "Vysota_potolkov": Vysota_potolkov,
        "Sanuzel": Sanuzel,
        "Vid_iz_okon": Vid_iz_okon,
        "Remont": Remont,
        "Musoroprovod": Musoroprovod,
        "Kolichestvo_liftov": Kolichestvo_liftov,
        "Tip_doma": Tip_doma,
        "Tip_perekrytij": Tip_perekrytij,
        "Parkovka": Parkovka,
        "Otoplenie": Otoplenie,
        "Balkonlodzhija": Balkonlodzhija,
        "Otdelka": Otdelka,
        "Etazhnost_zdanija": Etazhnost_zdanija,
        "Kolvo_komnat": Kolvo_komnat,
    }

    st.write(
        f"""### Данные клиента:\n
    1) Адрес: {dict_data['Address']}
    2) Метро: {dict_data['Metro']}
    3) Время до метро: {dict_data['Vremja_do_metro']}
    4) Общая площадь: {dict_data['Obschaja_ploschad']}
    5) Жилая площадь: {dict_data['Zhilaja_ploschad']}
    6) Площадь кухни: {dict_data['Ploschad_kuhni']}
    7) Этаж: {dict_data['Etazh']}
    8) Год постройки: {dict_data['God_postrojki']}
    9) Тип жилья: {dict_data['Tip_zhilja']}
    10) Высота потолков: {dict_data['Vysota_potolkov']}
    11) Санузел: {dict_data['Sanuzel']}
    12) Вид из окон: {dict_data['Vid_iz_okon']}
    13) Ремонт: {dict_data['Remont']}
    14) Мусоропровод: {dict_data['Musoroprovod']}
    15) Количество лифтов: {dict_data['Kolichestvo_liftov']}
    16) Тип дома: {dict_data['Tip_doma']}
    16) Тип перекрытий: {dict_data['Tip_perekrytij']}
    17) Парковка: {dict_data['Parkovka']}
    18) Отопление: {dict_data['Otoplenie']}
    19) Балкон/лоджия: {dict_data['Balkonlodzhija']}
    20) Отделка: {dict_data['Otdelka']}
    21) Этажность здания: {dict_data['Etazhnost_zdanija']}
    22) Кол-во комнат: {dict_data['Kolvo_komnat']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Предсказать")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f" {output}")
        st.success("Завершено")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Предсказать")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["Цена за квадрат(Предсказание)"] = output.json()["prediction"]
        st.write(
            data_.head().style.format(
                {"Цена за квадрат(Предсказание)": "{:.0f}"}, precision=0
            )
        )
