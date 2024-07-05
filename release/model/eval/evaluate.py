import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

if TYPE_CHECKING:
    # forward declaration to avoid circular dependencies
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop


class Evaluator:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder

    def _scale_inverse(self, X):
        return np.exp(X) - 1

    def evaluate(self):
        mapes = []
        mses = []
        bias = []
        if self.holder.hp.mode.val_mode_on:
            for product_id in self.holder.hp.shops.product_ids:
                trues = self._scale_inverse(self.holder.product_storage[product_id].ts_predict["demand"])
                avg_price = self.holder.product_storage[product_id].avg_price
                preds = self.holder.demand_metrics[product_id].price_storage[avg_price]["demand"]

                mape_ = mape(trues, preds)
                mse_ = mse(trues, preds)
                bias_ = np.asarray(trues) - np.asarray(preds)

                mapes.append(mape_)
                mses.append(mse_)
                bias.append(bias_)

                # print(f"for {product_id}: mape: {mape_}, mse: {mse_}")
                # print(trues, preds)
        else:
            print("Only use in val_mode_on")

        return {"mape": np.nanmean(mapes), "mse": np.nanmean(mses), "bias": np.nanmean(bias)}
