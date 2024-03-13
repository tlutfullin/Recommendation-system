import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def modify_column_data_type(col_name: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a pandas dataframe and a column name as input and returns the modified data type of the column.

    Parameters:
    col_name (pd.DataFrame): The pandas dataframe containing the column whose data type needs to be modified.

    Returns:
    pd.DataFrame: The modified data type of the column.

    Raises:
    Exception: If the data type of the column cannot be modified, it raises an exception.

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
        dtype_dict = {
            col: modify_column_data_type(df[col]) for col in df.columns
        }
        df = df.astype(dtype_dict)

        return df

    except FileNotFoundError as e:
        raise e

    finally:
        print(f"Data loaded from {path}")


def remove_null_data(df: pd.DataFrame, count_null: float = 0.8) -> pd.DataFrame:

    pass


df = load_data("data/raw/train_df.csv")

print(df.info())

