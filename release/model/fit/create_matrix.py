import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # forward declaration to avoid circular dependencies
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop


class MatrixGenerator:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder

    def _extract_stacked_features(self, X: pd.DataFrame):
        """
        Формирует матрицу стэковых фичей для одного датафрейма
        для создания предсказаний.

        Parameters
        ----------
        X : np.ndarray
            Матрица всех фичей
        days_as_features : int
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
        days_as_features = self.holder.hp.tech_features.days_as_features
        n_features_stacked = self.holder.hp.tech_features.n_features_stacked

        y = X.iloc[days_as_features:, 0]
        X = X.iloc[:, :n_features_stacked]

        X = np.apply_along_axis(
            lambda x: np.lib.stride_tricks.sliding_window_view(x, days_as_features), arr=X,
            axis=0)[:-1]

        X = np.reshape(X, newshape=(
            X.shape[0], X.shape[1] * X.shape[2]), order='F')

        return X, y

    def _prepare_train_matrices(self, X: pd.DataFrame):
        """
        Добавляет к матрице стэковых фичей фичи нового дня.

        Parameters
        ----------
        X : np.ndarray
            Матрица всех фичей
        days_as_features : int
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

        days_as_features = self.holder.hp.tech_features.days_as_features
        X_train_stacked, y_train_demand = self._extract_stacked_features(X)
        X_train_new_tick_features = np.asarray(X.iloc[days_as_features:, 1:])
        X_train_final = np.hstack((X_train_stacked, X_train_new_tick_features))

        return X_train_final, y_train_demand

    def _create_products_matrix(self):
        """
        Единственное предназначение функции – объединить временные ряды для отдельных товаров в одну обучающую матрицу.

        Parameters
        ----------
        holder: Preprocessor
            Наполнение класса описано при декларации.
        n: int
            Кол-во дней для добавления
        value: int
            Средняя цена за последние 7 дней.
            В последствии будет использована как базовая цена [от нее будем делать отклонение +- 2 std]
            для будущих дней для предсказаний.
            Заполняет пустые цены.

        Return
        ------
        X_train_final_united: np.ndarray[\sum_i^n(len_i) - K * n, n_features]
            len_i – длина i-го датафрейма, K – кол-во дней как фичи в модель, n – кол-во товаров.
        y_train_demand_united : np.ndarray[\sum_i^n(len_i) - K * n, 1]
            Датафрейм с demand как таргет.
        """

        product_storage = self.holder.product_storage
        k0 = list(product_storage.keys())[0]
        X_train_0 = self.holder.product_storage[k0].ts
        X_train, y_train = self._prepare_train_matrices(X_train_0)

        for k, v in product_storage.items():
            if k == k0:
                continue

            X_train_i, y_train_i = self._prepare_train_matrices(v.ts)
            X_train = np.vstack((X_train, X_train_i))
            y_train = np.append(y_train, y_train_i)

        return X_train, y_train

    def _add_product_feature(self, X):
        """
        Единственное предназначение функции – добавить OneHotEncoding колонки, которае будут отвечать за отдельный товар.

        Parameters
        ----------
        X: np.ndarray[\sum_i^n(len_i) - K * n, n_features]
            len_i – длина i-го датафрейма, K – кол-во дней как фичи в модель, n – кол-во товаров.
        holder: Preprocessor
            Наполнение класса описано при декларации.

        Return
        ------
        X: np.ndarray[\sum_i^n(len_i) - K * n, n_features + n]
            Объединенные датафрейм.
        """

        product_storage = self.holder.product_storage
        days_as_features = self.holder.hp.tech_features.days_as_features
        keys = list(product_storage.keys())

        products_periods = [self.holder.product_storage[k].length - days_as_features
                            for k in keys]

        assert set(product_storage.keys()) == set(self.holder.hp.shops.product_ids)
        feature = np.zeros((sum(products_periods), 1))
        cum_sum = 0
        for i, k in enumerate(keys):
            curr_period = products_periods[i]
            feature[cum_sum: cum_sum + curr_period, 0] = self.holder.hp.shops.le_product_ids.transform([k])[0]
            cum_sum += curr_period
        return np.hstack((X, feature))

    def get_matrix(self):
        X_train, y_train = self._create_products_matrix()

        X_train = self._add_product_feature(X_train)
        return X_train, y_train
