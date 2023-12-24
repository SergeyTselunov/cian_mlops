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


class InsuranceCustomer(BaseModel):
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
def prediction_input(customer: InsuranceCustomer):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            customer.Address,
            customer.Metro,
            customer.Vremja_do_metro,
            customer.Obschaja_ploschad,
            customer.Zhilaja_ploschad,
            customer.Ploschad_kuhni,
            customer.Etazh,
            customer.God_postrojki,
            customer.Tip_zhilja,
            customer.Vysota_potolkov,
            customer.Sanuzel,
            customer.Vid_iz_okon,
            customer.Remont,
            customer.Musoroprovod,
            customer.Kolichestvo_liftov,
            customer.Tip_doma,
            customer.Tip_perekrytij,
            customer.Parkovka,
            customer.Otoplenie,
            customer.Balkonlodzhija,
            customer.Otdelka,
            customer.Etazhnost_zdanija,
            customer.Kolvo_komnat,
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
