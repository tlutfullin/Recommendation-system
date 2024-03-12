import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score



## ====================================== Общий макет графика для всех коэффициентов ============================================= ##
def plot_heatmap(matrix_data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_data, annot=True, cmap='RdYlGn',linewidths=0.2, fmt='.2f')
    plt.title(f"{title}", fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()


## ====================================== Коэффициенты V-Крамера  ============================================= ##

# подсчет коэффициентов V-Крамера для двух признаков
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# строим матрицу коэффициентов для всех категориальных признаков
def cramers_matrix(df):
    cols = df.columns
    n_cols = len(cols)
    corr_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            corr_matrix[i,j] = cramers_v(df[cols[i]], df[cols[j]])
    
    return pd.DataFrame(corr_matrix, index=cols, columns=cols)




## ================================== Коэффициенты меры информационной взаимосвязи (Mutual Information) ============================== ##


def mutual_information(x, y):
    return mutual_info_score(x, y)


def mutual_information_matrix(df):
    cols = df.columns
    n_cols = len(cols)
    mi_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            mi_matrix[i,j] = mutual_information(df[cols[i]], df[cols[j]])
    
    return pd.DataFrame(mi_matrix, index=cols, columns=cols)



## ====================================== Коэффициенты меры условной энтропии ============================================= ##

def conditional_entropy(x, y):
    # Строим таблицу сопряженности
    contingency_table = pd.crosstab(x, y)
    
    # Получаем общее количество наблюдений
    total_samples = contingency_table.sum().sum()
    
    # Вычисляем меру условной энтропии
    conditional_entropy = 0
    for col in contingency_table.columns:
        px = contingency_table[col].sum() / total_samples
        for row in contingency_table.index:
            py_given_x = contingency_table.loc[row, col] / contingency_table[col].sum()
            pxy = contingency_table.loc[row, col] / total_samples
            if pxy > 0:
                conditional_entropy += pxy * np.log2(py_given_x / px)
    
    return -conditional_entropy



def conditional_entropy_matrix(df):
    # Создаем пустую матрицу условной энтропии
    n_cols = len(df.columns)
    cond_entropy_matrix = np.zeros((n_cols, n_cols))

    # Вычисляем меру условной энтропии для каждой пары признаков
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i != j:  # Исключаем диагональные элементы
                cond_entropy_matrix[i, j] = conditional_entropy(df[col1], df[col2])

    return pd.DataFrame(cond_entropy_matrix, index=df.columns, columns=df.columns)



