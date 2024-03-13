import pandas as pd

from prepare_data import load_data


def clean_data(data: pd.DataFrame, threshold: float=0.8) -> pd.DataFrame:
    """
    This function cleans a dataframe by removing duplicate rows, columns, and columns with a single value.

    Parameters:
    data (pd.DataFrame): The dataframe to be cleaned.
    threshold (float, optional): The proportion of non-null values in a column required to keep the column. Defaults to 0.8.

    Returns:
    pd.DataFrame: The cleaned dataframe.

    Raises:
    Exception: If an error occurs during cleaning.
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
        constant_name = data.columns[data.nunique()==1]
        data = data.drop(constant_name)

        return data

    except Exception as e:
        print(e)
        return data

