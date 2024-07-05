import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder
from collections import defaultdict


class DemandPredictor:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder

    def scale_inverse(self, X):
        return np.exp(X) - 1

    def _update_prices_dict(self, df, product_id, premium):

        storage = self.holder.product_storage[product_id]
        pure_profit = storage.pure_profit

        df["demand"] = self.scale_inverse(df["demand"])
        df["price"] = df["price"]

        df["profit"] = df["demand"] * (pure_profit + premium)
        df["gmv"] = df["demand"] * df["price"]

        self.holder.demand_metrics[product_id].price_storage[premium + storage.avg_price] = df

    def _create_df(self, product_id, premium):
        ts = self.holder.product_storage[product_id].ts
        ts_predict = self.holder.product_storage[product_id].ts_predict.copy()
        ts_predict.iloc[:, 1] = ts_predict.iloc[:, 1] + premium
        return pd.concat([ts, ts_predict], axis=0)

    def _predict_single_price(self, product_id, premium):
        df = self._create_df(product_id, premium)

        # получаем предикты
        preds_df = self.holder.model_wrapper.predict(df, product_id)

        # добавляем новую цену в словарь цен
        self._update_prices_dict(preds_df[-self.holder.hp.tech_features.prediction_period:], product_id,
                                 premium)

    def _get_boundaries(self, product_id):
        storage: 'MetaHolder' = self.holder.product_storage[product_id]
        min_price, max_price = storage.min_max_price["min_price"], storage.min_max_price["max_price"]
        return min_price, max_price

    def scale(self, X):
        return np.log1p(X)

    def _predict_multiple_prices(self, product_id):
        price_std = self.holder.product_storage[product_id].price_std
        num = 7
        assert num % 2 == 1  # for metrics calculation
        std = 1
        for premium in np.linspace(-std * price_std, std * price_std, num=num):
            self._predict_single_price(product_id, premium)

    def _predict_multiple_offers(self):
        for product_id in self.holder.hp.shops.product_ids:
            print(f"_predict_multiple_offers for :{product_id}")
            self._predict_multiple_prices(product_id)

    def update_demand_metaholder(self):
        self._predict_multiple_offers()


class PricePicker:
    """
    Данный класс используется для выбора наилучшей цены из prices_dict на основе выбранной стратегии:
    Возможные стратегии:
    1. maximize_profit
        Стратегия максимизации прибыли

    Parameters для __init__
    ----------

    Атрибуты класса
    ------

    Методы класса
    ------
    def _loop(d: pd.DataFrame, f: callable)
        Пробегается по всем ценам и выбирает лучшую на основании f

        Parameters
        ----------
        d: pd.DataFrame
            prices_dict

        f: callable
            функция, по которой брать argmax будет браться argmax

        Return
        ------
        best_price: float
            Наилучшая цена

    """

    def __init__(self, holder: 'OffersHolder') -> None:
        self.holder = holder

    def _loop(self, d: pd.DataFrame, f: callable):
        "Функция для подсчета определенной метрики и выбора наилучшей по критерию f: callable среди всех цен"

        best_price = -float('inf')
        best_metric = -float('inf')

        for price, d in d.items():
            metric = f(d)
            if metric > best_metric:
                best_metric = metric
                best_price = price
        return best_price

    def _maximize_profit(self, price_metrics):
        def f(x): return (x["profit"]).sum()

        return self._loop(price_metrics, f)

    def apply_strategy(self, price_metrics):
        return self._maximize_profit(price_metrics)


class BestPricePredictor:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder
        self.prices_predictor = DemandPredictor(self.holder)
        self.price_picker = PricePicker(self.holder)

    def _get_boundaries(self, product_id):
        storage: 'MetaHolder' = self.holder.product_storage[product_id]
        min_price, max_price = storage.min_max_price["min_price"], storage.min_max_price["max_price"]
        return min_price, max_price

    def _predict_best_prices(self):
        # для каждого продукта
        for product_id in self.holder.hp.shops.product_ids:
            demand_metrics = self.holder.demand_metrics[product_id].price_storage
            recommended_price = self.price_picker.apply_strategy(demand_metrics)

            min_price, max_price = self._get_boundaries(product_id)
            if min_price is not None and max_price is not None:
                recommended_price = np.clip(recommended_price, a_min=min_price, a_max=max_price)

            self.holder.demand_metrics[product_id].recommended_price = recommended_price

    def _collect_best_prices(self):
        d = defaultdict(dict)
        for product_id in self.holder.hp.shops.product_ids:
            shop_name, offer_id = product_id.split("_", 1)
            shop_id = self.holder.hp.shops.shop_name_to_id[shop_name]
            d[shop_id][offer_id] = self.holder.demand_metrics[product_id].recommended_price
        return d

    def predict(self):
        self._predict_best_prices()
        return self._collect_best_prices()
