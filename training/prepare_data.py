import pandas as pd


# Преобразование типов данных
def modify_column_data_type(col_name: pd.DataFrame) -> pd.DataFrame:
    """
    Эта функция принимает в качестве входных данных кадр данных pandas и имя столбца и возвращает измененный тип данных столбца.

    Параметры:
    col_name (pd.DataFrame): кадр данных pandas, содержащий столбец, тип данных которого необходимо изменить.

    Возврат:
    pd.DataFrame: измененный тип данных столбца.

    Поднимает:
    Исключение: если тип данных столбца не может быть изменен, возникает исключение.

    """
    try:
        if col_name.dtypes == "float64":
            return "float32"

        if col_name.dtypes == "int64":
            return "int32"
        else:
            return col_name.dtypes

    except Exception as e:
        print(e)


# загрузка данных из CSV файла
def load_data(path: str) -> pd.DataFrame:
    """
    This function loads data from a CSV file into a pandas dataframe.

    Parameters:
    path (str): The path to the CSV file containing the data.

    Returns:
    pd.DataFrame: The pandas dataframe containing the data.

    Raises:
    FileNotFoundError: If the file cannot be found, it raises a FileNotFoundError.
    """

    try:

        df = pd.read_csv(path)

        # Оцениваем объем памяти, используемый DataFrame
        memory_usage = df.memory_usage(deep=True).sum()
        print(f"Memory usage before: {memory_usage//(1024**2):.3f} MB")

        dtype_dict = {
            col: modify_column_data_type(df[col]) for col in df.columns
        }
        df = df.astype(dtype_dict)

        # Оцениваем объем памяти, используемый DataFrame после преобразования типов данных
        memory_usage = df.memory_usage(deep=True).sum()
        print(f"Memory usage after: {memory_usage/(1024**2):.3f} MB")

        return df

    except FileNotFoundError as e:
        raise e

    finally:
        print(f"Data loaded from {path}")
