"""
Программа: Frontend часть проекта
Версия: 1.0
"""
import os
import time
import yaml
import streamlit as st
from src.data.get_data import get_dataset, load_data
from src.plotting.charts import pair_plot, reg_heat_plot, kde_plot, bar_plot, box_plot
from src.train.training import start_training, last_metrics, diff_metrics, charts
from src.evaluate.evaluate import evaluate_input, evaluate_from_file
from src.train.cross_validation import cross_val

st.set_option("deprecation.showPyplotGlobalUse", False)
CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://www.asiacryptotoday.com/wp-content/uploads/2023/01/Cian.jpg",
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project:  Apartment price prediction 🏠💸")
    st.write(
        """
        В настоящее время найти квартиру себе по душе с идеальным расположением, 
        площадью и инфраструктурой – одна из самых сложных и в то же время актуальных задач.
        Нужно обратить внимание на множество факторов – из каких материалов и как давно построен дом,
        в каком он состоянии, сколько в нем этажей, его расположение и, конечно, его стоимость.
         
        Поэтому наша задача построить модель, которая будет предсказывать цену за квадрат на недвижимость 
        в Москве на основе разных факторов."""
    )

    # name of the columns
    st.write(
        """
        Базу данных взял на основе парсинга данных моего проекта: https://github.com/SergeyTselunov/Cian
        
        ### Описание полей 
        - Название - Название объявления
        - Адрес - Адрес квартиры
        - Метро - Ближайшая станция метро
        - Время до метро - Время до ближайшей станции метро (мин)
        - Цена - Цена за квартиру (руб)
        - Цена за квадрат - Цена за квадратный метр (руб) (Целевой признак)
        - Общая площадь - Общая площадь (кв. метры)
        - Жилая площадь - Жилая площадь (кв. метры)
        - Площадь кухни - Площадь кухни (кв. метры)
        - Этаж - Номер этажа из всех этажей дома 
        - Год постройки - Год постройки дома
        - Тип жилья - Вторичка или новостройка 
        - Высота потолков - Высота потолков (метры) 
        - Санузел - Количество и тип санузлов
        - Вид из окон - Вид из окон
        - Ремонт - Тип римонта
        - Строительная серия - Строительная серия
        - Мусоропровод - Наличие мусоропровода 
        - Количество лифтов - количество лифтов в доме
        - Тип дома - Тип дома
        - Тип перекрытий - Тип перекрытий
        - Парковка - Вид парковки
        - Подъезды - Количество подъездов
        - Отопление - Тип отопления
        - Аварийность - Аварийность
        - Газоснабжение - Наличие газоснабжения
        - Балкон/лоджия - Количество и тип балкона/лоджии
        - Год сдачи - Год сдачи квартир в доме
        - Дом - Сдан или нет
        - Отделка - Тип отделки 
        - Кол-во комнат - количество комнат в квартире.
        - Широта метро, долгота метро - координаты метро 
        - Широта дома, долго дома - координаты дома 
        - Расстояние до метро, расстояние до центра - Расстояния до ближайшего метро и центра Москвы соответственно.
        - Округ - Округ Москвы в котором находится квартира."""
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Разведочный анализ данных📈📊")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["cian_full_path"])
    st.write(data.head())

    time.sleep(1)

    # # plotting with checkbox
    dist_metro_price = st.sidebar.checkbox("Расстояние до метро - Цена за квадрат")
    dist_center_price = st.sidebar.checkbox("Расстояние до центра - Цена за квадрат")
    floor_price = st.sidebar.checkbox("Этаж - Цена за квадрат")
    parking_price = st.sidebar.checkbox("Парковка - Цена за квадрат")
    type_house_price = st.sidebar.checkbox("Тип дома - Цена за квадрат")
    #
    time.sleep(1)

    if dist_metro_price:
        st.pyplot(
            pair_plot(
                data=data,
                data_x="Расстояние до метро",
                data_y="Цена за квадрат",
                title="Расстояние до метро - Цена за квадрат",
            )
        )
        st.markdown(
            "Между признаками есть корреляция, но совсем небольшая. Зато мы видим, что большинство квартир продаются "
            "близко к метро, так как в Москве на данный момент сильно развит метрополитен."
        )

    if dist_center_price:
        st.pyplot(
            reg_heat_plot(
                data=data,
                data_x="Расстояние до центра",
                data_y="Цена за квадрат",
                title="Расстояние до центра - Цена за квадрат",
            )
        )
        st.markdown(
            "Между признаками есть достаточно сильная корреляция. Цена за 1 квадратный метр ближе к центру "
            "города выше."
        )
    if floor_price:
        fl_floor = data[
            (data["Этаж"] == 1) | (data["Этаж"] == data["Этажность здания"])
        ]["Цена за квадрат"]
        other_floor = data.drop(fl_floor.index)[["Цена за квадрат"]]
        st.pyplot(
            kde_plot(
                data_1=fl_floor,
                data_2=other_floor,
                title="Этаж - Цена за квадрат",
                leg=["Первый и последний этаж", "Остальные этажи"],
            )
        )
        st.markdown(
            "Цена за квадрат квартир на первом и последнем этаже ниже чем у остальных. В первом случае из-за подвала "
            "и высокого уровня шума с улицы. В втором случае из-за возможных протечек крыши."
        )
    if parking_price:
        st.pyplot(
            bar_plot(
                data=data,
                data_x="Парковка",
                data_y="Цена за квадрат",
                title="Парковка - Цена за квадрат",
            )
        )
        st.markdown(
            "Квартиры в домах с подземной или многоуровневой парковкой дороже чем остальные. Связано с тем, "
            "что машина в любое время года находится в теплом, закрытом помещении. И нет необходимости зимой тратить "
            "время на прогрев машины и очистку от снега в зимнее время."
        )
    if type_house_price:
        st.pyplot(
            box_plot(
                data=data,
                data_x="Цена за квадрат",
                data_y="Тип дома",
                title="Тип дома - Цена за квадрат",
            )
        )
        st.markdown(
            "Квартиры в монолитно-кирпичных домах дороже чем в других. У этих домов самый высокий срок эксплуатации, "
            "хорошая звуко-изоляция, и эта технология позволяет возводить дома разных форм."
        )


def run():
    """
    Состояние сеанса для Streamlit.
    Флаг запуска обучения модели.
    """
    st.session_state.run = True


def tog():
    """
    Состояние сеанса для Streamlit.
    Флаг смены параметров для обучения модели.
    """
    st.session_state.tog = True


def check():
    """
    Состояние сеанса для Streamlit.
    Флаг кнопки для кросс-валидации.
    """
    st.session_state.check = True


def clear():
    """
    Состояние сеанса для Streamlit.
    Очистка результатов.
    """
    st.session_state.new_metrics = None
    st.session_state.cross = None
    st.session_state.overfit = None
    st.session_state.dis = False


def training():
    """
    Тренировка модели
    """
    st.markdown("# Тренировка модели LightGBM🔁")

    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Инициализация состояний сеанса
    if "run" not in st.session_state:
        st.session_state.run = False
        st.session_state.new_metrics = None
        st.session_state.tog = False
        st.session_state.check = False
        st.session_state.cross = None
        st.session_state.overfit = None
        st.session_state.dis = False

    if "old_metrics" not in st.session_state:
        st.session_state.old_metrics = last_metrics(config=config, visible=False)
    st.markdown("##")
    main_1 = st.empty()
    st.markdown("##")
    main_2 = st.empty()
    st.markdown("##")
    body = main_1.container()

    # Создание колонок
    with body:
        col1, col2 = st.columns(2)
    body2 = main_2.container()

    with body2:
        col3, col4 = st.columns(2)

    metric = st.empty()
    metric_old = metric.container()
    metric_2 = st.empty()
    metric_new = metric_2.container()

    # Создание кнопки для смены параметров
    with col2:
        on = st.toggle(
            "Обучение на лучших параметрах",
            disabled=st.session_state.dis,
        )
        if on:
            st.session_state.tog = True
        else:
            st.session_state.tog = False

    # Смена параметров модели
    if st.session_state.tog:
        with metric_old:
            st.markdown("### Метрики последней обученной модели")
            last_metrics(config=config, visible=True)
        endpoint = config["endpoints"]["training_top"]
    else:
        endpoint = config["endpoints"]["train"]

    # Проверка на запуск обучения для отображения кнопок.
    if not st.session_state.run and not st.session_state.check:
        if (
            st.session_state.new_metrics is not None
            or st.session_state.cross is not None
        ):
            col1.button("Запустить обучение еще раз", on_click=run)
        else:
            col1.button("Запустить обучение", on_click=run)
        # Смена параметров модели
    else:
        main_1.empty()
        st.markdown("##")
        main_2.empty()
        time.sleep(1)
    # Кросс-валидация
    if st.session_state.check:
        main_1.empty()
        main_2.empty()
        time.sleep(1)
        st.session_state.cross, st.session_state.overfit = cross_val(
            endpoint=config["endpoints"]["cross_training"]
        )
        st.session_state.check = False
        st.rerun()
    # Обучение модели
    if st.session_state.run:
        st.session_state.old_metrics = last_metrics(config=config, visible=False)
        st.session_state.new_metrics = start_training(endpoint=endpoint)
        st.session_state.run = False
        st.session_state.dis = True
        st.rerun()
    # Отображение результатов обучения.
    if st.session_state.new_metrics is not None and st.session_state.cross is None:
        if not st.session_state.tog:
            with metric_new:
                st.write("### Сравнение метрик новой модели")
                diff_metrics(
                    new_metrics=st.session_state.new_metrics,
                    old_metrics=st.session_state.old_metrics,
                )
        charts(config)
        with body2:
            st.success("Обучение завершено")
        col3.button("Проверка на кросс-валидации", on_click=check)
        col4.button("Очистить результаты", on_click=clear)
    # Отображение результатов кросс-валидации.
    elif st.session_state.cross is not None:
        with metric_new:
            st.write("##")
            st.write("### Метрики на кросс-валидации")
            diff_metrics(
                new_metrics=st.session_state.cross,
                old_metrics=st.session_state.new_metrics,
            )
        with body2:
            st.success("Проверка завершена")
        st.session_state.cross = None
        st.write("## Переобучение:", st.session_state.overfit, "%")
        col4.button("Очистить результаты", on_click=clear)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Предсказание по заполненным данным💸🏘️")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["train"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Предсказание из файла📁")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Разведочный анализ данных": exploratory,
        "Тренировка модели LightGBM": training,
        "Предсказание по заполненным данным": prediction,
        "Предсказание из файла": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
