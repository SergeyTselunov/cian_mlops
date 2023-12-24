"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""
import joblib
import yaml
import os

from ..data.create_data import create_data
from ..data.get_data import get_dataset
from ..data.split_dataset import split_train_test, split_val
from ..train.train import find_optimal_params, train_model
from ..transform.feature_engineering import pipeline_features
from ..transform.geo_code import pipeline_geo
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str, top_params: bool = False) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param top_params: Флаг для тренировки на лучших параметрах.
    :param config_path: Путь до файла с конфигурациями
    :return: None
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # get data
    train_data = get_dataset(dataset_path=preprocessing_config["cian_path"])

    # preprocessing
    train_data = pipeline_preprocess(
        data=train_data, flg_evaluate=False, **train_config
    )
    create_data(data=train_data, dataset_path=preprocessing_config["cian_clean_path"])

    # geo
    train_data = pipeline_geo(data=train_data, **preprocessing_config)
    create_data(data=train_data, dataset_path=preprocessing_config["cian_full_path"])

    # features_engineering
    train_data = pipeline_features(data=train_data, flg_evaluate=False, **train_config)

    # split data
    df_train, df_test = split_train_test(dataset=train_data, **preprocessing_config)

    # split val data
    df_train_, df_val = split_val(dataset=train_data, **preprocessing_config)
    # find optimal params
    if not top_params:
        study = find_optimal_params(
            data_train=df_train,
            data_test=df_test,
            **train_config,
        )
    else:
        study = joblib.load(os.path.join(config["train"]["study_path"]))
    # train with optimal params
    clf = train_model(
        data_train=df_train,
        data_test=df_test,
        data_train_=df_train_,
        data_val=df_val,
        study=study,
        top_params=top_params,
        target=train_config["target_column"],
        metric_path=train_config["metrics_path"],
        params_path=train_config["params_path"],
    )

    # save result (study, model)
    joblib.dump(clf, os.path.join(train_config["model_path"]))
    joblib.dump(study, os.path.join(train_config["study_path"]))
