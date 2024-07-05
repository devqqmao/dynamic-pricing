import numpy as np
import pandas as pd
from datetime import date

from dynamic_pricing.pipeline.features.feature_collector import FeatureCollector
from marketplace.models import Shop
# from typing import List
# import string

def get_specific_product_ids():
    """
    Единственное назначение функции – возвращать offer_ids, на которых будет модель обучаться.

    Parameters
    ----------

    Return
    ------
    List[offer_ids]
    """

    offer_ids = ['PC-TWCS-UTP-RJ45-RJ45-C5e-15M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-0.5M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-10M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-5M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-30M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C6-15M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-2M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-25M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C5e-20M-G',
                 'PC-TWCS-UTP-RJ45-RJ45-C6-10M-G']

    return offer_ids


def prepare_dataset(df):
    """
    Функция принимает датасет и делает изменения специфичные для !!конкретного датасета и конкретных товаров!!.

    Parameters
    ----------
    df : pd.DataFrame
        Только что загруженный pd.DataFrame

    Return
    ------
    df : pd.DataFrame
        Со специфичными изменениями
    """

    # selects columns
    df = df.rename({"quantity": "Demand",
                    "marketing_seller_price": "Price",
                    "sales_profit": "Profit",
                    "offer_id": "Product",
                    }, axis=1)

    # fills np.NaN demand
    df.Demand.fillna(0, inplace=True)

    # resample with frequency
    df = df.groupby('Product').resample('D').agg(
        {'Demand': 'mean',
         'Price': 'mean',
         'Profit': 'mean',
         }).reset_index(level=0)

    # fill in missing values with bfill
    df.set_index(["Product", df.index], inplace=True)
    df.Price.replace(0, np.nan, inplace=True)
    df.Profit.replace(0, np.nan, inplace=True)
    df = df.groupby(level=0).bfill()
    df.reset_index(0, inplace=True)
    df.fillna(0, inplace=True)

    # sort index
    df.sort_index(inplace=True)

    # choose columns
    df = df.loc[:, ["Demand", "Price",
                    "Profit",
                    "Product"]]

    return df


def trim_products(df: pd.DataFrame, product_ids):
    """
    Функция принимает датасет:
    1. Обрезает дату (т.к. последние n-дней – np.NaN)
    2. Выбирает конкретные товары

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм после prepare_dataset

    product_ids: [string]
        Offer_id, которые необходимо оставить.

    Return
    ------
    df : pd.DataFrame
        Со специфичными изменениями
    """

    from datetime import datetime, timedelta

    d = datetime.today() - timedelta(days=6)
    df = df.loc[pd.Timestamp(2023, 9, 1):d, :]

    df_trimmed = df[(df['Product'].isin(product_ids))]
    return df_trimmed


def choose_product(df, id):
    """
    Функция принимает датасет и отдает конкретный продукт.

    Parameters
    ----------
    df : pd.DataFrame

    product_ids: [string]
        Offer_id, которые необходимо оставить.

    Return
    ------
    df : pd.DataFrame
        Со специфичными изменениями
    """

    ts = df[df['Product'] == id]
    ts = ts.drop(["Product"], axis=1, inplace=False)
    return ts


def load_dataset():
    """
    Функция загружает датасет с определенными параметрами.

    Parameters
    ----------

    Return
    ------
    df : pd.DataFrame
    """

    # define params
    shop_id = "479580b8-30b1-4ace-9da1-77649e3c39ee"

    # define dates
    start_date = date(2024, 1, 1)
    end_date = date(2024, 2, 1)

    # load dataset
    shop = Shop(id=shop_id)
    df = FeatureCollector(shop=shop, start_date=start_date,
                          end_date=end_date).collect()

    # parse, set, sort index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", drop=True, inplace=True)
    df.sort_index(inplace=True)

    return df


def create_dataset(self):
    """
    Функция вызывается из Holder.init().
    1. Загружает датафрейм
    2. Вызывает функцию prepare_dataset

    Parameters
    ----------
    self
        Instance of holder

    Return
    ------
    df : pd.DataFrame
    """

    # loads dataset
    df = load_dataset()
    # prepares dataset
    df = prepare_dataset(df)

    # loads product_ids if not passed
    if self.product_ids is None:
        self.product_ids = get_specific_product_ids()

    # sets number of products
    self.n_products = len(self.product_ids)

    # selects products
    df_selected = trim_products(df, self.product_ids)

    return df_selected
