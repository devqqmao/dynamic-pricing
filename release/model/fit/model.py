import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from typing import TYPE_CHECKING
import pickle
import pickle
import os

if TYPE_CHECKING:
    # forward declaration to avoid circular dependencies
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder


class ModelWrapper:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder
        self.feature_importances_ = None
        self.model_path = self.holder.hp.local.catboost_model_path
        self.model_meta_path = self.holder.hp.local.model_meta_path
        self.feature_importances_path = self.holder.hp.local.catboost_feature_importances_path
        self.model_catboost = None

    def _save_model(self, model: CatBoostRegressor):
        if not os.path.exists(self.model_meta_path):
            os.makedirs(self.model_meta_path)

        model.save_model(self.model_path)

        self.feature_importances_ = model.feature_importances_

        with open(self.feature_importances_path, 'wb') as handle:
            pickle.dump(self.feature_importances_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_model(self):

        model = CatBoostRegressor()
        model.load_model(self.model_path)

        with open(self.feature_importances_path, 'rb') as handle:
            self.feature_importances_ = pickle.load(handle)

        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        n, m = X_train.shape
        product_ids_feature_id = m - 1
        product_ids_qty = len(self.holder.hp.shops.product_ids)

        X_train: pd.DataFrame = pd.DataFrame(X_train, columns=list(range(m)), dtype=np.float64)
        X_train.iloc[:, product_ids_feature_id] = X_train.iloc[:, product_ids_feature_id].astype(int)
        y_train = pd.Series(y_train)

        if self.holder.hp.local.load_model:
            model = self._load_model()
            return model

        model = CatBoostRegressor(iterations=5000,
                                  random_seed=42,
                                  early_stopping_rounds=100,
                                  learning_rate=0.1,

                                  allow_writing_files=False,
                                  per_float_feature_quantization=[
                                      f'{product_ids_feature_id}:border_count={product_ids_qty}'],
                                  one_hot_max_size=product_ids_qty,
                                  cat_features=[product_ids_feature_id]
                                  )

        if self.holder.hp.local.retrain_model:
            model.fit(X_train, y_train,
                      verbose=False,
                      plot=False
                      )
            if self.holder.hp.local.save_model:
                self._save_model(model)

        return model

    def _update_features(self, idx, df, predicted_demand):
        # update demand inplace
        df.iloc[-self.holder.hp.tech_features.prediction_period + idx, 0] = predicted_demand

    def _collect_features(self, idx, df, product_id):
        # считаем стэковые фичи
        K = self.holder.hp.tech_features.days_as_features
        N = self.holder.hp.tech_features.prediction_period
        S = self.holder.hp.tech_features.n_features_stacked

        features = np.asarray(
            df[-(K + N) + idx: -N + idx].iloc[:, :S]).flatten(order="A")

        # добавляем не стэковые
        features = np.append(features, np.asarray(
            df.iloc[-N + idx][1:]))

        # добавляем фичи продукта

        features = pd.DataFrame(features.reshape(1, -1), dtype=np.float64)
        features[len(features.columns)] = self.holder.hp.shops.le_product_ids.transform([product_id])[0]

        return features

    def _predict_single_tick(self, i, df, offer_id):
        features = self._collect_features(i, df, offer_id)
        predicted_demand = self.model_catboost.predict(features)
        self._update_features(i, df, predicted_demand)

    def _predict_multiple_ticks(self, df, product_id):
        # для N дней
        for i in range(self.holder.hp.tech_features.prediction_period):
            self._predict_single_tick(i, df, product_id)

        return df

    def predict(self, df, product_id):
        return self._predict_multiple_ticks(df, product_id)
