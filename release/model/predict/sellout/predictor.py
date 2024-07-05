import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
import math

from collections import namedtuple, defaultdict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder


class PricePickerSellout:
    def __init__(self, offers_holder: 'OffersHolder') -> None:
        self.offers_holder = offers_holder

    def _calculate_total_added_capital(self, capital_values):
        percentage = 0.15 / 365
        total_added_capital = 0
        capital_by_day = []
        for i in range(len(capital_values)):
            cap = capital_values[i] * (1 + percentage) ** i
            total_added_capital += cap
            capital_by_day.append(total_added_capital)
        return capital_by_day, total_added_capital - sum(capital_values)

    def _calculate_costs(self, period_demand, sellout_period, meta_holder):
        total_costs = 0
        period_demand_total = period_demand.sum()
        days_until_sellout = math.ceil(sellout_period * 7)

        curr_stocks = period_demand_total
        costs_for_one = meta_holder.volume_costs * meta_holder.volume
        costs_by_day = []
        for i in range(days_until_sellout):
            curr_stocks -= period_demand[i % len(period_demand)]
            metric = curr_stocks * costs_for_one
            costs_by_day.append(metric)
            total_costs += metric

        return costs_by_day, total_costs

    def _loop(self, product_id):
        "Функция для подсчета определенной метрики и выбора наилучшей по критерию f: callable среди всех цен"
        meta_holder: 'MetaHolder' = self.offers_holder.product_storage[product_id]

        best_price = -float('inf')
        best_metric = -float('inf')

        storage = dict()
        for price, df in self.offers_holder.demand_metrics[product_id].price_storage.items():

            capital_by_day, period_capital = self._calculate_total_added_capital(df["gmv"])

            demand_by_day = df["demand"]
            period_demand = demand_by_day.sum()

            period_sellout = np.float64(meta_holder.stocks) / period_demand

            storage_costs_by_day, period_storage_costs = self._calculate_costs(demand_by_day, period_sellout,
                                                                               meta_holder)

            profit_by_day = df["profit"]
            period_profit = profit_by_day.sum()
            metric = period_profit + period_capital - period_storage_costs

            sub_metrics = dict(zip(
                ['capital_by_day', 'demand_by_day', 'storage_costs_by_day', 'profit_by_day',
                 'period_capital', 'period_demand', 'period_storage_costs', 'period_profit'],
                [capital_by_day, demand_by_day, storage_costs_by_day, profit_by_day,
                 period_capital, period_demand, period_storage_costs, period_profit]))

            storage[price] = {"metric": metric, "period_sellout": period_sellout,
                              "sub_metrics": sub_metrics}

            if metric > best_metric:
                best_metric = metric
                best_price = price

        return best_price, storage

    def apply_strategy(self, product_id):
        return self._loop(product_id)


class SelloutPredictor:
    def __init__(self, offers_holder: 'OffersHolder'):
        self.holder = offers_holder
        self.price_picker = PricePickerSellout(self.holder)

    def _get_boundaries(self, product_id):
        storage: 'MetaHolder' = self.holder.product_storage[product_id]
        min_price, max_price = storage.min_max_price["min_price"], storage.min_max_price["max_price"]
        return min_price, max_price

    def _update_product_recommended_price(self):
        # для каждого продукта
        for product_id in self.holder.hp.shops.product_ids:
            recommended_price, storage = self.price_picker.apply_strategy(product_id)

            min_price, max_price = self._get_boundaries(product_id)
            if min_price is not None and max_price is not None:
                recommended_price = np.clip(recommended_price, a_min=min_price, a_max=max_price)

            self.holder.sellout_metrics[product_id].recommended_price = recommended_price
            self.holder.sellout_metrics[product_id].price_storage = storage

    def _collect_best_prices(self):
        d = defaultdict(dict)
        for product_id in self.holder.hp.shops.product_ids:
            shop_name, offer_id = product_id.split("_", 1)
            shop_id = self.holder.hp.shops.shop_name_to_id[shop_name]
            d[shop_id][offer_id] = self.holder.sellout_metrics[product_id].recommended_price
        return d

    def predict(self):
        self._update_product_recommended_price()
        return self.holder, self._collect_best_prices()
