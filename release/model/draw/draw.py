from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

pd.options.plotting.backend = "plotly"
import os
import warnings

warnings.filterwarnings('ignore')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # forward declaration to avoid circular dependencies
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder


class Artist:

    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder

    def scale_inverse(self, X):
        return np.exp(X) - 1

    def draw_timeseries(self, ts, label):
        demand_median, price_median = ts.iloc[:, :2].median(axis=0)

        plt.plot(ts["demand"], label=f'demand_{label}')
        plt.plot(ts["price"] * ((demand_median / price_median) * 2),
                 label=f'price_{label}')
        plt.legend()

    def draw_predictions(self, product_id):
        index = self.holder.product_storage[product_id].ts_predict.index

        preds_dict = self.holder.demand_metrics[product_id].price_storage
        avg_price = self.holder.product_storage[product_id].avg_price

        plt.figure(figsize=(16, 12))
        plt.grid(visible=True)
        plt.xticks(index, [date.strftime('%m-%d') for date in index], rotation=45)

        if self.holder.hp.mode.val_mode_on:
            trues = self.scale_inverse(self.holder.product_storage[product_id].ts_predict["demand"])
            plt.plot(pd.Series(trues, index=index), '-go', label=round(avg_price, 2), linewidth=1)

        for price, df in preds_dict.items():
            plt.plot(pd.Series(df["demand"], index=index), label=round(price, 2), linewidth=2)

        plt.legend(loc='best', fontsize="20")
        plt.show()

    def draw_sellout_curve(self, product_id):
        preds_dict = self.holder.demand_metrics[product_id].price_storage

        plt.figure(figsize=(16, 12))
        plt.grid(visible=True)

        for price, df in preds_dict.items():
            stocks = self.holder.product_storage[product_id].stocks
            predicted_demand = df["demand"]
            index = self.holder.product_storage[product_id].ts.index[-1]
            index = [index]

            stocks_remainders = list()
            stocks_remainders.append(stocks)

            i = 0
            while stocks > 0:
                stocks -= predicted_demand[i % self.holder.hp.tech_features.prediction_period]
                stocks_remainders.append(stocks)
                i += 1

                # Add an extra day to the index inside the loop
                index.append(index[-1] + pd.DateOffset(days=1))
            plt.title(product_id)
            plt.xticks(index, [date.strftime('%m-%d') for date in index], rotation=45)
            plt.plot(pd.Series(stocks_remainders, index=index), label=round(price, 2), linewidth=2)

        plt.legend(loc='best', fontsize="20")
        plt.show()

    def draw_feature_importances(self):

        columns = self.holder.product_storage[self.holder.hp.shops.product_ids[0]].ts_predict.columns
        cols = []

        [cols.extend(c) for c in
         [["_".join([str(i), col]) for i in range(self.holder.hp.tech_features.days_as_features)]
          for col in columns[:self.holder.hp.tech_features.n_features_stacked]]]

        def handler(x): return "LT_" + x

        cols.extend(list(map(handler, columns[1:])))
        cols = list(cols) + ["product_id"]

        plt.figure(figsize=(32, 10))
        plt.grid(visible=True)
        plt.xticks(np.arange(len(cols)), cols, rotation=45)
        plt.bar(cols, self.holder.model_wrapper.feature_importances_)
        plt.show()


class SelloutReport:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder

    def scale_inverse(self, X):
        return np.exp(X) - 1

    def save_metric_to_csv(self, product_id, metric, csv_path):
        preds_dict = self.holder.sellout_metrics[product_id].price_storage
        data = []

        for price, storage in preds_dict.items():
            m = storage["sub_metrics"][metric]
            data.append({'price': round(price, 2), metric: m})

        df = pd.DataFrame(data)
        csv_file = os.path.join(csv_path, f'{metric}.csv')
        df.to_csv(csv_file, index=False)

    def save_graph_as_image(self, product_id, metric, image_path):
        if metric == 'stocks':
            self.draw_stocks(product_id, image_path)
        else:
            self.draw_metric(product_id, metric, image_path)

    def draw_stocks(self, product_id, image_path=None):
        preds_dict = self.holder.demand_metrics[product_id].price_storage

        plt.figure(figsize=(16, 12))
        plt.grid(visible=True)

        for price, df in preds_dict.items():
            stocks = self.holder.product_storage[product_id].stocks
            predicted_demand = df["demand"]
            index = self.holder.product_storage[product_id].ts.index[-1]
            index = [index]

            stocks_remainders = list()
            stocks_remainders.append(stocks)

            i = 0
            while stocks > 0:
                stocks -= predicted_demand[i % self.holder.hp.tech_features.prediction_period]
                stocks_remainders.append(stocks)
                i += 1

                # Add an extra day to the index inside the loop
                index.append(index[-1] + pd.DateOffset(days=1))
            plt.title(product_id)
            plt.xticks(index, [date.strftime('%m-%d') for date in index], rotation=45)
            plt.plot(pd.Series(stocks_remainders, index=index), label=round(price, 2), linewidth=2)

        plt.legend(loc='best', fontsize="20")
        if image_path is not None:
            image_file = os.path.join(image_path, f'stocks.png')
            plt.savefig(image_file)
        plt.show()

    def print_metric(self, product_id, metric):
        preds_dict = self.holder.sellout_metrics[product_id].price_storage

        for price, storage in preds_dict.items():
            m = storage["sub_metrics"][metric]
            print(f"price: {round(price, 2)}, metric: {m}")

    def draw_metric(self, product_id, metric, image_path=None):
        index = self.holder.product_storage[product_id].ts.index[-1]

        preds_dict = self.holder.sellout_metrics[product_id].price_storage

        # get max_len
        max_len = -float('inf')
        for price, storage in preds_dict.items():
            cur_len = len(storage["sub_metrics"][metric])
            if cur_len > max_len:
                max_len = cur_len
        # get indices
        indices = []
        for i in range(max_len):
            # Add one day to the current index
            new_index = index + timedelta(days=i)
            indices.append(new_index)

        # fill to max_len
        for price, storage in preds_dict.items():
            cur_len = len(storage["sub_metrics"][metric])
            for i in range(max_len - cur_len):
                storage["sub_metrics"][metric].append(0)
            assert (len(storage["sub_metrics"][metric]) == max_len)

        plt.figure(figsize=(16, 12))
        plt.grid(visible=True)
        plt.xticks(indices, [date.strftime('%m-%d') for date in indices], rotation=45)

        for price, storage in preds_dict.items():
            plt.plot(pd.Series(storage["sub_metrics"][metric], index=indices), label=round(price, 2),
                     linewidth=2)

        plt.legend(loc='best', fontsize="20")
        if image_path is not None:
            image_file = os.path.join(image_path, f'{metric}.png')
            plt.savefig(image_file)
        plt.show()
