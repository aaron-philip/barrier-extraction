import pandas as pd
import numpy as np
import os

def load_experimental_data(dirname: str, expfiles: list[str]) -> list[pd.DataFrame]:
    """
    Load experimental data files.
    Returns list of DataFrames with columns ['Ecm','Sigma','Error'].
    """
    exp = []
    for file in expfiles:
        df = pd.read_csv(os.path.join(dirname, file), sep="\s+", names=['Ecm','Sigma','Error'])
        if df['Ecm'].iloc[0] > df['Ecm'].iloc[-1]:
            df = df.iloc[::-1]
        exp.append(df)
    return exp


def get_data_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract numpy arrays E_data, Sigma_data, Sigma_err from DataFrame.
    """
    return df['Ecm'].to_numpy(), df['Sigma'].to_numpy(), df['Error'].to_numpy()