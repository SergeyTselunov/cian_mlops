"""
Программа: Отрисовка графиков
Версия: 1.0
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


plt.rcParams["figure.figsize"] = 15, 7
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_palette("viridis")


def pair_plot(data: pd.DataFrame, data_x: str, data_y: str, title: str):
    """
    Отрисовка графика PairPlot
    :param data_y: Признак.
    :param data_x: Признак.
    :param data: Датасет.
    :param title: Название графика.
    """
    sns.pairplot(data[[data_x, data_y]], height=5)
    plt.title(title, y=2.1, x=-0.1)


def reg_heat_plot(data: pd.DataFrame, data_x: str, data_y: str, title: str):
    """
    Отрисовка regplot вместе с heatplot.
    :param data: Датасет.
    :param data_x: Признак по Х
    :param data_y: Признак по Y
    :param title: Название графика.
    """

    fig, axes = plt.subplots(1, 2)
    sns.regplot(x=data[data_x], y=data[data_y], ax=axes[0], marker="o")
    sns.heatmap(data=(data[[data_x, data_y]]).corr(), annot=True)
    plt.title(title, x=-0.1, fontsize=20)
    plt.ylabel(data_y, fontsize=18)
    plt.xlabel(data_x, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def kde_plot(data_1: pd.Series, title: str, data_2: pd.Series = None, leg: list = None):
    """
    Отрисовка kdeplot.
    :param data_1: Признак из датасета
    :param data_2: Признак из датасета
    :param title: Название графика
    :param leg: Список наименований для легенды
    """
    fig = plt.figure()
    sns.kdeplot(data_1, color="r", common_norm=False, palette="viridis")
    sns.kdeplot(data_2, common_norm=False, palette="viridis")
    plt.title(title, fontsize=20)
    plt.legend(labels=leg)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def bar_plot(data: pd.DataFrame, data_x: str, data_y: str, title: str):
    """
    Отрисовка barplot.
    :param data: датасет
    :param data_x: Признак по Х
    :param data_y: Признак по Y
    :param title: Название графика
    """
    fig = plt.figure()
    sns.barplot(data=data, x=data_x, y=data_y, palette="viridis")
    plt.title(title, fontsize=20)
    plt.ylabel(data_y, fontsize=18)
    plt.xlabel(data_x, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def box_plot(data: pd.DataFrame, data_x: str, data_y: str, title: str):
    """
    Отрисовка boxplot
    :param data: датасет
    :param data_x: Признак по Х
    :param data_y: Признак по Y
    :param title: Название графика
    """
    fig = plt.figure()
    sns.boxplot(data, y=data_y, x=data_x, orient="h", palette="viridis")
    plt.title(title, fontsize=20)
    plt.ylabel(data_y, fontsize=18)
    plt.xlabel(data_x, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig
