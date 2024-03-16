from operator import index
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from typing import Union, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sqlalchemy import column


# создадим новый признак представляющий собой сумму, среднее, дисперсию всех признаков за исключением некоторых
# бинаризируем признаки на несколько частей/долей


def feature_engineering(
    data: pd.DataFrame,
    columns_exclude: list = ["search_id", "target"],
) -> pd.DataFrame:
    """
    Эта функция выполняет преобразование данных во входном кадре данных, включая:
    1. Удаление мультиколлинеарных объектов (feature_3 и Feature_77) и сохранение только одного из них.
    2. Создание новых функций путем суммирования, усреднения и расчета стандартного отклонения оставшихся функций.
    3. Объединение определенных функций в определенное количество категорий.
    4. Преобразовывает определенные столбцы(которые были отнесены к категориальным в EDA) в категориальный тип

    Аргументы:
        data: (pd.DataFrame): входной кадр данных
        columns_exclude (список, необязательно): список столбцов, которые необходимо исключить из преобразования. По умолчанию ["search_id", "target"].

    Возврат:
        pd.DataFrame: преобразованный фрейм данных.
    """

    try:

        # удалим мультиколлинеарные признаки(feature_3 и feature_4) и (feature_77, feature_78), оставив только один из них
        if "feature_3" in data.columns:
            data = data.drop(columns=["feature_3"])

        if "feature_77" in data.columns:
            data = data.drop(columns=["feature_77"])

        # выбираем столбцы, которые будут учтены для создания новых функций путем суммирования, усреднения и расчета дисперсии
        included_columns = [col for col in data.columns if col not in columns_exclude]

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

        if all(feature in data.columns for feature in features_to_binarize):

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

        # преобразуем признаки в категориальный тип
        cat_columns = [
            "search_id",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
            "feature_6",
            "feature_7",
            "feature_8",
            "feature_9",
            "feature_10",
            "feature_11",
            "feature_12",
            "feature_13",
            "feature_14",
            "feature_15",
            "feature_61",
        ]

        for column in cat_columns:
            if column in data.columns:
                data[column] = data[column].astype("category")

        return data

    except Exception as e:
        print(e)
        return data


def encoding(
    data: pd.DataFrame,
    cat_features: Union[List[str], None] = None,
    encoder_method: Union[List[str], None] = ["ordinal", "one_hot"],
    save_path: Union[str, None] = "encoders.pkl",
) -> pd.DataFrame:

    """
    Эта функция кодирует категориальные особенности данного кадра данных, используя указанные методы кодирования.

    Аргументы:
        data: (pd.DataFrame): входной кадр данных
        cat_features: (Union[List[str], None]): список категориальных функций для кодирования. Если None, все категориальные функции в кадре данных будут закодированы.
        encoder_method: (Union[List[str], None]): список используемых методов кодирования. Допустимые параметры: «ordinal» и «one_hot». Если None, будет использоваться OrdinalEncoder.
        save_path: (Union[str, None]): путь для сохранения обученных кодировщиков. Если None, кодировщики не будут сохранены.

    Возвращает:
        pd.DataFrame: закодированный кадр данных.

    Возникает:
        Исключение: если неправильно написан метод кодирования или во время обработки возникает ошибка.
    """
    
    # из датафрейма отбираем все категориальные данные
    if cat_features is None:
        cat_features = data.select_dtypes(include=["category", "object"]).columns.tolist()

    try:

        # создаем словарь для хранения обученных кодировщиков
        encoders = {}

        # создаем пустой датафрейм, который будет содержать закодированные данные
        encoded_data = pd.DataFrame(index=data.index)

        for feature in cat_features:

            # проверяем какой метода кодирования использовать
            if "ordinal" in encoder_method:
                # для признаков, которые не попали в обучение будет ставится (unknown_value=-1)
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,  # устанавливает закодированное значение для неизвестных категорий.
                    encoded_missing_value=-2,  # для пропущенных(None) меток будет стоять значение -2
                    dtype=np.int64,
                )

            elif "one_hot" in encoder_method:
                encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.int64)
            else:
                raise ValueError("Unsupported encoding method")

            # Преобразование категориального признака
            encoded_feature = encoder.fit_transform(data[[feature]])

            # сохраняем обученный кодировщик в словарь
            encoders[feature] = encoder

            # преобразуем разряженную матрицу к DataFrame
            if sparse.issparse(encoded_feature):
                if "one_hot" in encoder_method:
                    encoded_feature_names = [
                        f"{feature}_{x}" for x in encoder.get_feature_names_out()
                    ]
                    encoded_feature_df = pd.DataFrame(
                        encoded_feature.toarray(), index=data.index, columns=encoded_feature_names
                    )

                else:
                    encoded_feature_df = pd.DataFrame(
                        encoded_feature.toarray(), index=data.index, columns=[feature]
                    )

            else:
                encoded_feature_df = pd.DataFrame(
                    encoded_feature, index=data.index, columns=[feature]
                )

            # Добавляем закодированный признак к общему DataFrame
            encoded_data = pd.concat([encoded_data, encoded_feature_df], axis=1)

        # Конкатенируем закодированные признаки с исходными данными
        data = pd.concat([data, encoded_data], axis=1)

        # удаляем исходные категориальные переменные
        data.drop(columns=cat_features, inplace=True)

        # сохраняем кодировщик в pickle формат
        with open(save_path, "wb") as file:
            pickle.dump(encoders, file)

        return data

    except Exception as e:
        print(e)
        return data