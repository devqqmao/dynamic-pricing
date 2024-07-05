import numpy as np
import pandas as pd
from collections import defaultdict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.utils import log_df
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop
    from dynamic_pricing.ml.release.model.preprocess.extract_features import FeatureExtractor
    from dynamic_pricing.ml.release.model.load.load_dataset import DatasetLoader
from collections import defaultdict


class DemandMetricsHolder:
    def __init__(self):
        self.price_storage = defaultdict(pd.DataFrame)
        self.recommended_price = defaultdict(dict)


class SelloutMetricsHolder:
    def __init__(self):
        self.price_storage = defaultdict(pd.DataFrame)
        self.recommended_price = defaultdict(dict)


class MetaHolder:
    def __init__(self):
        self.offer_id = None
        self.shop_id = None
        self.volume_costs = None
        self.volume = None
        self.stocks = None
        self.pure_costs = None
        self.min_max_price = None
        self.price = None

        self.avg_price = None
        self.avg_price_vector = None
        self.length = None
        self.ts_predict = None
        self.ts = None
        self.pure_profit = None
        self.price_std = None


class OffersHolder:

    def __init__(self, hyper_parameters: 'HyperParameters', feature_extractor: 'FeatureExtractor',
                 dataset_loader: 'DatasetLoader') -> None:
        self.hp = hyper_parameters
        self.fe = feature_extractor
        self.dl = dataset_loader

        self.dl.init_loader()

        self.product_storage = defaultdict(MetaHolder)
        self.demand_metrics = defaultdict(DemandMetricsHolder)
        self.sellout_metrics = defaultdict(SelloutMetricsHolder)

        self.model_wrapper = None

    def _choose_product(self, df, product_id: str):
        ts = df[df['product'] == product_id]
        ts = ts.drop(["product"], axis=1, inplace=False)
        return ts

    def _append_prediction_length(self, ts, prices):
        n = self.hp.tech_features.prediction_period

        last_date = ts.index[-1]
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n, freq='D')
        new_data = pd.DataFrame(index=new_dates)

        df = pd.concat((ts, new_data))
        df.iloc[-n:, 1] = prices
        return df

    def _add_features(self, product_ts, shop_name):
        product_ts = self.fe.add_features(product_ts, shop_name)
        return product_ts

    def _split_ts(self, df):

        first = df.iloc[:-self.hp.tech_features.prediction_period]
        second = df.iloc[-self.hp.tech_features.prediction_period:]
        return first, second

    def _apply_delay(self, df: pd.DataFrame):
        return df.iloc[:-self.hp.shops.delay_timeline]

    def _update_product_storage_base(self, product_id, ts, ts_predict, avg_price, avg_price_vector, shop_id, offer_id):

        self.product_storage[product_id].ts = ts
        self.product_storage[product_id].ts_predict = ts_predict
        self.product_storage[product_id].length = len(ts)
        self.product_storage[product_id].avg_price = avg_price
        self.product_storage[product_id].avg_price_vector = avg_price_vector

        self.product_storage[product_id].shop_id = shop_id
        self.product_storage[product_id].offer_id = offer_id

        self.product_storage[product_id].pure_profit = self.product_storage[product_id].avg_price - \
                                                       self.product_storage[product_id].pure_costs
        self.product_storage[product_id].price_std = np.std(ts["price"])

    def _update_product_storage_meta(self, shop_name, product_id, offer_id):
        # for demand
        self.product_storage[product_id].min_max_price = self.dl.get_meta(shop_name, "min_max_prices")[offer_id]
        self.product_storage[product_id].pure_costs = self.dl.get_meta(shop_name, "pure_costs")[offer_id]

        # for sellout
        if self.hp.prediction_goal.sellout:
            self.product_storage[product_id].stocks = self.dl.get_meta(shop_name, "stocks")[offer_id]
            self.product_storage[product_id].volume = self.dl.get_meta(shop_name, "volume")[offer_id]
            self.product_storage[product_id].volume_costs = self.dl.get_meta(shop_name, "volume_costs")[offer_id]

        # self.product_storage[product_id].price = self.dl.get_meta(shop_name, "prices")[offer_id]

    def _scale(self, product_ts: pd.DataFrame):
        product_ts["demand"] = np.log1p(product_ts["demand"])
        return product_ts

    def _get_price_vector(self, product_ts):
        if not self.hp.mode.val_mode_on:
            avg_price_vector = product_ts['price'][-self.hp.tech_features.prediction_period:]
        else:
            avg_price_vector = [None]
        return avg_price_vector

    def _get_price_avg(self, product_ts):
        mean_price = np.nanmean(product_ts.iloc[-self.hp.tech_features.prediction_period:, 1])
        return mean_price

    def extract_and_preprocess(self, df, shop_name):
        """
        инвариант, после product_id все магазины не различимы
        """

        for product_id in self.hp.shops.product_ids:
            if not product_id.startswith(shop_name):
                continue

            product_ts = self._choose_product(df, product_id)
            product_ts = self._apply_delay(product_ts)

            avg_price = self._get_price_avg(product_ts)
            price_vector = self._get_price_vector(product_ts)

            if not self.hp.mode.val_mode_on:
                product_ts = self._append_prediction_length(product_ts, avg_price)

            product_ts = self._scale(product_ts)
            # log_df(product_ts)
            product_ts = self._add_features(product_ts, shop_name)

            ts, ts_predict = self._split_ts(product_ts)

            self.product_storage[product_id] = MetaHolder()
            shop_id, offer_id = product_id.split('_', 1)
            self._update_product_storage_meta(shop_name, product_id, offer_id)
            self._update_product_storage_base(product_id, ts, ts_predict, avg_price, price_vector,
                                              shop_id, offer_id)
