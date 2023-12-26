import warnings

import pandas as pd

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
import optuna
import uvicorn

from src.evaluate.evaluate import pipeline_evaluate
from src.pipelines.pipeline import pipeline_training
from src.train.metrics import load_metrics
from src.train.cross_val import pipeline_cross

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class ApartmentMoscow(BaseModel):
    """
    Признаки для получения результатов модели
    """

    Address: str
    Metro: str
    Vremja_do_metro: int
    Obschaja_ploschad: float
    Zhilaja_ploschad: float
    Ploschad_kuhni: float
    Etazh: int
    God_postrojki: int
    Tip_zhilja: object
    Vysota_potolkov: float
    Sanuzel: int
    Vid_iz_okon: object
    Remont: object
    Musoroprovod: object
    Kolichestvo_liftov: int
    Tip_doma: object
    Tip_perekrytij: object
    Parkovka: object
    Otoplenie: object
    Balkonlodzhija: int
    Otdelka: object
    Etazhnost_zdanija: int
    Kolvo_komnat: int


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/train_top_params")
def training_top():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH, top_params=True)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/cross_training")
def cross_validation():
    """
    Проверка модели на кросс валидации
    """
    overfit = pipeline_cross(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH, cross_val=True)
    return {"metrics": metrics, "overfit": overfit}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(apartment: ApartmentMoscow):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            apartment.Address,
            apartment.Metro,
            apartment.Vremja_do_metro,
            apartment.Obschaja_ploschad,
            apartment.Zhilaja_ploschad,
            apartment.Ploschad_kuhni,
            apartment.Etazh,
            apartment.God_postrojki,
            apartment.Tip_zhilja,
            apartment.Vysota_potolkov,
            apartment.Sanuzel,
            apartment.Vid_iz_okon,
            apartment.Remont,
            apartment.Musoroprovod,
            apartment.Kolichestvo_liftov,
            apartment.Tip_doma,
            apartment.Tip_perekrytij,
            apartment.Parkovka,
            apartment.Otoplenie,
            apartment.Balkonlodzhija,
            apartment.Otdelka,
            apartment.Etazhnost_zdanija,
            apartment.Kolvo_komnat,
        ]
    ]

    cols = [
        "Адрес",
        "Метро",
        "Время до метро",
        "Общая площадь",
        "Жилая площадь",
        "Площадь кухни",
        "Этаж",
        "Год постройки",
        "Тип жилья",
        "Высота потолков",
        "Санузел",
        "Вид из окон",
        "Ремонт",
        "Мусоропровод",
        "Количество лифтов",
        "Тип дома",
        "Тип перекрытий",
        "Парковка",
        "Отопление",
        "Балкон/лоджия",
        "Отделка",
        "Этажность здания",
        "Кол-во комнат",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(
        config_path=CONFIG_PATH, dataset=data, flg_input=True
    )[0]
    price = round((predictions * data["Общая площадь"][0]) / 1e6, 1)
    pred = round(predictions / 1e6, 3)
    result = f"""
    Цена за квадрат: {pred}млн.\n
    Цена: {price}млн.
    """
    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=8000)
