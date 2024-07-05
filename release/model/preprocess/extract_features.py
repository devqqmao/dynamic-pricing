import numpy as np
import pandas as pd
from dynamic_pricing.ml.release.hyper_parameters import HyperParameters

from dynamic_pricing.ml.release import utils


class FeatureExtractor:
    def __init__(self, hyper_parameters: 'HyperParameters') -> None:
        self.hp = hyper_parameters

    def _add_features_datetime(self, df: pd.DataFrame):
        dates = pd.DataFrame(index=df.index)
        dates["date"] = pd.to_datetime(df.index)
        df["day_of_month_sin"] = np.sin(
            (2 * np.pi * dates['date'].dt.day) / dates['date'].dt.days_in_month[0])
        df["day_of_month_cos"] = np.cos(
            (2 * np.pi * dates['date'].dt.day) / dates['date'].dt.days_in_month[0])

        # because starts with 0
        df["day_of_week_sin"] = np.sin(
            (2 * np.pi * (dates['date'].dt.day_of_week + 1)) / 7)
        df["day_of_week_cos"] = np.cos(
            (2 * np.pi * dates['date'].dt.day_of_week + 1) / 7)

        df["day_of_year_sin"] = np.sin(
            (2 * np.pi * dates['date'].dt.dayofyear) / 365)
        df["day_of_year_cos"] = np.cos(
            (2 * np.pi * dates['date'].dt.dayofyear) / 365)

        return df

    def _add_features_shop(self, df, shop_name):
        df["shop"] = shop_name
        return df

    def _add_features_category(self):
        pass

    def _add_features_day_since_launched(self):
        pass

    def add_features(self, df, shop_name):
        print(f"enter: add_features: {shop_name}")
        df = self._add_features_datetime(df)

        return df
