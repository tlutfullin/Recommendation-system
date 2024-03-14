import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def feature_engineering(
    data: pd.DataFrame,
    columns_exclude: list = ["search_id", "target"],
) -> pd.DataFrame:
    
    """
    Эта функция выполняет преобразование данных во входном кадре данных, включая:
    1. Удаление мультиколлинеарных объектов (feature_3 и Feature_77) и сохранение только одного из них.
    2. Создание новых функций путем суммирования, усреднения и расчета стандартного отклонения оставшихся функций.
    3. Объединение определенных функций в определенное количество категорий.

    Аргументы:
        data: (pd.DataFrame): входной кадр данных
        columns_exclude (список, необязательно): список столбцов, которые необходимо исключить из преобразования. По умолчанию ["search_id", "target"].

    Возврат:
        pd.DataFrame: преобразованный фрейм данных.
    """


    try:

        # удалим мультиколлинеарные признаки(feature_3 и feature_77), оставив только один из них
        if "feature_3" in data.columns:
            data = data.drop(columns=["feature_3"])

        if 'feature_77' in data.columns:
            data = data.drop(columns=["feature_77"])

        
        # выбираем столбцы, которые будут учестены для создания новых функций путем суммирования, усреднения и расчета дисперсии
        included_columns = [col for col in df.columns if col not in columns_exclude]

        # названия столбцов для созданных новых признаков 
        name_columns = ["feature_sum", "feature_mean", "feature_std"]
        
        # создадим новые признаки на основе: суммы, средних, дисперсии старых признаков
        data[name_columns[0]] = data[included_columns].sum(axis=1)
        data[name_columns[1]] = data[included_columns].mean(axis=1)
        data[name_columns[2]] = data[included_columns].std(axis=1)

        # признаки которые будут бинаризоваться и соответственно их количество бинов
        features_to_binarize = [
            "feature_24",
            "feature_25",
            "feature_26",
            "feature_44",
            "feature_68",
        ]

        if features_to_binarize in data.columns:


            # бинаризируем признаки
            data["discrete_feat_24"] = pd.qcut(
                x=data[features_to_binarize[0]],
                q=8,
                labels=[f"category_{i}" for i in range(8)],
            )
            data["discrete_feat_25"] = pd.qcut(
                x=data[features_to_binarize[1]],
                q=7,
                labels=[f"category_{i}" for i in range(7)],
            )
            data["discrete_feat_26"] = pd.qcut(
                x=data[features_to_binarize[2]],
                q=5,
                labels=[f"category_{i}" for i in range(5)],
            )
            data["discrete_feat_44"] = pd.qcut(
                x=data[features_to_binarize[3]],
                q=4,
                labels=[f"category_{i}" for i in range(4)],
            )
            data["discrete_feat_68"] = pd.qcut(
                x=data[features_to_binarize[4]],
                q=4,
                labels=[f"category_{i}" for i in range(4)],
            )

        return data

    except Exception as e:
        print(e)
        return data
    


    