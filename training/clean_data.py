import pandas as pd
from prepare_data import load_data


def clean_data(data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Эта функция очищает фрейм данных, удаляя повторяющиеся строки, столбцы и столбцы с одним значением.

    Параметры:
    data (pd.DataFrame): кадр данных, который необходимо очистить.
    threshold (float, необязательно): доля ненулевых значений в столбце, необходимая для сохранения столбца. По умолчанию 0,8.

    Возврат:
    pd.DataFrame: очищенный фрейм данных.

    Вызывает:
    Исключение: если во время очистки возникает ошибка.
    """
    try:
        # Сохраняем количество строк и столбцов  DataFrame
        num_rows = data.shape[0]
        num_columns = data.shape[1]

        # Удаляем повторяющиеся строки по значению
        data = data.drop_duplicates()

        # Удаляем повторяющиеся столбцы по значению
        data = data.T.drop_duplicates().T

        # Удаляем строки, в которых более threshold значений являются NaN
        data = data.dropna(axis=1, thresh=int(threshold * num_rows))
 
        # Удаляем столбцы, в которых больше threshold значений являются NaN
        data = data.dropna(axis=0, thresh=int(threshold * num_columns))

        # Удаляем константные признаки
        constant_name = data.columns[data.nunique() == 1]
        data = data.drop(constant_name)

        return data

    except Exception as e:
        print(e)
        return data


def remove_outliers(
    data: pd.DataFrame, columns: None, threshold: float = 40, q: float = 0.75
) -> pd.DataFrame:
    """
    Эта функция удаляет строки с указанным количеством выбросов в каждом столбце кадра данных на основе межквартильного диапазона (IQR).

    Параметры:
    data (pd.DataFrame): кадр данных, содержащий строки с выбросами.
    columns (нет, необязательно): список столбцов, которые необходимо проверить на наличие выбросов. Если нет, будут проверены все столбцы.
    threshold (с плавающей запятой, необязательно): пороговое значение количества выбросов. Если количество выбросов в столбце превышает это значение, строка будет удалена. По умолчанию 40.
    q (с плавающей запятой, необязательно): квантиль, используемый для расчета IQR. По умолчанию 0,75.

    Возврат:
    pd.DataFrame: кадр данных с удаленными аномальными строками.

    Вызывает:
    Исключение: если во время удаления выбросов возникает ошибка.
    """
    try:
        if columns is None:
            columns = data.columns

        if threshold is None:
            threshold = data.shape[1] * 0.5

        # Определение маски выбросов для каждого столбца
        mask = pd.DataFrame()
        for column in columns:
            # подсчёт значений для метода IQR
            Q1 = data[column].quantile(1 - q)
            Q3 = data[column].quantile(q)
            IQR = Q3 - Q1
            upper, lower = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR

            # найденные выбросы для столбца
            outliers = (data[column] < lower) | (data[column] > upper)
            mask[column] = outliers

        # подсчёт количества выбросов в каждой строке
        outliers_sum = mask.sum(axis=1)

        # удаление строк с количеством выбросов, превышающим порог
        return data[outliers_sum < threshold]

    except Exception as e:
        print(e)
        return data
