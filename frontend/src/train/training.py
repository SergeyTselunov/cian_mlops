"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(endpoint: object):
    """
    Тренировка модели.
    :param endpoint: endpoint
    """

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")
    new_metrics = output.json()["metrics"]
    return new_metrics


def charts(config):
    """
    Вывод графиков optuna.
    :param config: Конфигурационный файл
    """
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)
    # plot study
    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)


def diff_metrics(new_metrics, old_metrics):
    """
    Вывод изменений метрик после обучения.
    :param new_metrics: Новые метрики
    :param old_metrics: Прошлые метрики
    """
    # diff metrics
    mae, mse, rmse, rmsle = st.columns(4)
    mae.metric(
        "MAE",
        new_metrics["MAE"],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",
        delta_color="inverse",
    )
    mse.metric(
        "MSE",
        new_metrics["MSE"],
        f"{new_metrics['MSE']-old_metrics['MSE']:.3f}",
        delta_color="inverse",
    )
    rmse.metric(
        "RMSE",
        new_metrics["RMSE"],
        f"{new_metrics['RMSE']-old_metrics['RMSE']:.3f}",
        delta_color="inverse",
    )
    rmsle.metric(
        "RMSLE",
        new_metrics["RMSLE"],
        f"{new_metrics['RMSLE']-old_metrics['RMSLE']:.3f}",
        delta_color="inverse",
    )
    r2_adj, mpe, mape, wape = st.columns(4)

    r2_adj.metric(
        "R2_adjusted",
        new_metrics["R2_adjusted"],
        f"{new_metrics['R2_adjusted']-old_metrics['R2_adjusted']:.3f}",
    )
    mpe.metric(
        "MPE_%",
        new_metrics["MPE_%"],
        f"{new_metrics['MPE_%']-old_metrics['MPE_%']:.3f}",
        delta_color="inverse",
    )
    mape.metric(
        "MAPE_%",
        new_metrics["MAPE_%"],
        f"{new_metrics['MAPE_%']-old_metrics['MAPE_%']:.3f}",
        delta_color="inverse",
    )
    wape.metric(
        "WAPE_%",
        new_metrics["WAPE_%"],
        f"{new_metrics['WAPE_%']-old_metrics['WAPE_%']:.3f}",
        delta_color="inverse",
    )


def last_metrics(config: dict, visible: bool = True):
    """
    Отображение метрик последнего обучения модели.
    :param config: Конфигурационный файл
    :param visible: Флаг видимости.
    :return:
    """
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {
            "MAE": 0,
            "MSE": 0,
            "RMSE": 0,
            "RMSLE": 0,
            "R2_adjusted": 0,
            "MPE_%": 0,
            "MAPE_%": 0,
            "WAPE_%": 0,
        }
    if visible:
        mae, mse, rmse, rmsle = st.columns(4)
        mae.metric(
            "MAE",
            old_metrics["MAE"],
        )
        mse.metric(
            "MSE",
            old_metrics["MSE"],
        )
        rmse.metric(
            "RMSE",
            old_metrics["RMSE"],
        )
        rmsle.metric(
            "RMSLE",
            old_metrics["RMSLE"],
        )
        r2_adj, mpe, mape, wape = st.columns(4)

        r2_adj.metric(
            "R2_adjusted",
            old_metrics["R2_adjusted"],
        )
        mpe.metric(
            "MPE_%",
            old_metrics["MPE_%"],
        )
        mape.metric(
            "MAPE_%",
            old_metrics["MAPE_%"],
        )
        wape.metric(
            "WAPE_%",
            old_metrics["WAPE_%"],
        )
    return old_metrics
