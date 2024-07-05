import numpy as np
import pandas as pd


def add_features_datetime(df):
    """
    Функция добавляет временные фичи к df.

    Parameters
    ----------
    df: pd.DataFrame
        df после препроцессинга

    Return
    ------
    df: pd.DataFrame
        df с добавленными фичами
    """

    dates = pd.DataFrame(index=df.index)
    dates["date"] = pd.to_datetime(df.index)

    df["DayOfMonth_sin"] = np.sin(
        (2 * np.pi * dates['date'].dt.day) / dates['date'].dt.days_in_month[0])
    df["DayOfMonth_cos"] = np.cos(
        (2 * np.pi * dates['date'].dt.day) / dates['date'].dt.days_in_month[0])

    # because starts with 0
    df["DayOfWeek_sin"] = np.sin(
        (2 * np.pi * (dates['date'].dt.day_of_week + 1)) / 7)
    df["DayOfWeek_cos"] = np.cos(
        (2 * np.pi * dates['date'].dt.day_of_week + 1) / 7)

    df["DayOfYear_sin"] = np.sin(
        (2 * np.pi * dates['date'].dt.dayofyear) / 365)
    df["DayOfYear_cos"] = np.cos(
        (2 * np.pi * dates['date'].dt.dayofyear) / 365)

    return df



def add_features(df):
    """
    Функция последовательно применяет трансформации к df, добавляя дополнительные фичи.

    Parameters
    ----------
    df: pd.DataFrame
        df после препроцессинга

    Return
    ------
    df: pd.DataFrame
        df с добавленными фичами
    """
    df = add_features_datetime(df)
    return df
