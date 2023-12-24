import streamlit as st
import requests


def cross_val(endpoint: object):
    """
    Проверка модели на кросс-валидации
    :param endpoint:  endpoint
    :return:
    """
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")
    cross_metrics = output.json()["metrics"]
    overfit = output.json()["overfit"]
    st.write("## Переобучение = ", overfit, "%")
    return cross_metrics, overfit
