import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def extract_stacked_features(X, K, n_features_stacked):
    """
    Формирует матрицу стэковых фичей для одного датафрейма
    для создания предсказаний.

    Parameters
    ----------
    X : np.ndarray
        Матрица всех фичей
    K : int
        Количество дней, которые необходимо брать как фичи
    n_features_stacked : int
        Количество стаковых фичей

    Return
    ------
    X : np.ndarray
        Новая матрица стэковых фичей
    y: np.ndarray
        Вектор таргетов
    """

    y = X.iloc[K:, 0]
    X = X.iloc[:, :n_features_stacked]

    X = np.apply_along_axis(
        lambda x: np.lib.stride_tricks.sliding_window_view(x, K), arr=X,
        axis=0)[:-1]
    X = np.reshape(X, newshape=(
        X.shape[0], X.shape[1] * X.shape[2]), order='F')

    return X, y


def prepare_train_matrices(X, K, n_features_stacked):
    """
    Добавляет к матрице стэковых фичей фичи нового дня.

    Parameters
    ----------
    X : np.ndarray
        Матрица всех фичей
    K : int
        Количество дней, которые необходимо брать как фичи
    n_features_stacked : int
        Количество стаковых фичей

    Return
    ------
    X : np.ndarray
        Новая матрица стэковых фичей с фичами нового дня
    y: np.ndarray
        Вектор таргетов
    """

    X_train_stacked, y_train_demand = extract_stacked_features(
        X, K, n_features_stacked)
    X_train_new_tick_features = np.asarray(X.iloc[K:, 1:])
    X_train_final = np.hstack((X_train_stacked, X_train_new_tick_features))

    return X_train_final, y_train_demand
