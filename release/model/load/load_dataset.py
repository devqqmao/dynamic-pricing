import pandas as pd
import json
import datetime

from typing import TYPE_CHECKING

import os

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop

"""
things to load:
1. dataset
2. min_max_prices
3. offer_ids
4. pure_costs
5. volume
6. volume_costs
7. stocks
"""


class DatasetLoaderServerUtils:
    def __init__(self):
        pass

    def load_dataset(self, shop_id: str, start_date, end_date):
        from marketplace.models import Shop
        from dynamic_pricing.pipeline.features.feature_collector import FeatureCollector
        try:
            df = FeatureCollector(shop=Shop(id=shop_id), start_date=start_date,
                                  end_date=end_date).collect()
        except Exception as e:
            print(f"Unsuccessful load_dataset for: {shop_id}")
            raise e
        return df

    def load_min_max_prices(self, shop_id: str, offer_id: str):
        from dynamic_pricing.models import DynamicPricing
        try:
            item = DynamicPricing.objects.filter(
                shop_id=shop_id, offer_id=offer_id).values("min_price", "max_price")[0]
        except:
            item = {"min_price": None, "max_price": None}
            print(f"Unsuccessful load_min_max_prices for: {shop_id}:{offer_id}")
        return item

    def load_pure_costs(self, shop_id, offer_id):
        from marketplace.models import Product
        try:
            item = Product.objects.filter(shop_id=shop_id, offer_id=offer_id).values("cost_price")[0][
                "cost_price"]
        except Exception as e:
            print(f"Unsuccessful load_min_max_prices for: {shop_id}:{offer_id}")
            item = None
        return item

    def load_costs(self, shop_id, offer_id, price):
        from marketplace.models import Product
        from dynamic_pricing.data_processing.profitability import CalcProfitability
        try:
            product = Product.objects.get(shop_id=shop_id, offer_id=offer_id)
            profitability = CalcProfitability(product.id, price).agg_calc()
            item = price - (profitability * price / 100)
        except Exception as e:
            print(f"Unsuccessful load_total_cost for: {shop_id}:{offer_id}")
            item = None
        return item

    def load_stocks(self, shop_id, offer_id):
        from marketplace.models import StocksHistory
        try:
            today = datetime.date.today()
            values = StocksHistory.objects.filter(shop=shop_id,
                                                  offer_id=offer_id,
                                                  date_time__date=today).all().values('warehouse', 'stocks')
            stocks_df = pd.DataFrame.from_records(values)
            stocks_df = stocks_df.groupby(['warehouse'], as_index=False).agg({
                'stocks': 'mean'
            })
            item = sum(stocks_df['stocks'])
        except Exception as e:
            print(f"Unsuccessful load_stocks for: {shop_id}:{offer_id}")
            print(e)
            item = None
        return item

    def load_volume(self, shop_id, offer_id):
        from marketplace.models import Product
        try:
            item = Product.objects.get(shop_id=shop_id, offer_id=offer_id).volume
        except Exception as e:
            print(f"Unsuccessful load_volume for: {shop_id}:{offer_id}")
            item = None
        return item

    def load_volume_cost(self, shop_id, offer_id):
        from marketplace.models import Product
        from dynamic_pricing.data_processing.profitability import CalcProfitability
        try:
            product = Product.objects.get(shop_id=shop_id, offer_id=offer_id)
            item = CalcProfitability(product.id, 0).calc_placement_cost()
        except Exception as e:
            print(f"Unsuccessful load_volume_cost for: {shop_id}:{offer_id}")
            print(e)
            item = None
        return item

    def load_offer_ids(self, shop_id):
        from marketplace.models import Shop
        try:
            item = Shop.objects.get(id=shop_id).products.filter(is_active=True).values_list('offer_id', flat=True)
        except Exception as e:
            print(f"Unsuccessful load_offer_ids for: {shop_id}")
            print(e)
            item = None
        return item

    def set_date(self, hp, meta: 'Shop'):
        if hp.mode.train_on_dates:
            start_date = meta.start_date
            end_date = meta.end_date
        else:
            start_date = hp.shops.default_start_date
            end_date = hp.shops.default_end_date
        return start_date, end_date


class DatasetLoaderServer(DatasetLoaderServerUtils):
    def __init__(self, hyper_parameters: 'HyperParameters') -> None:
        super().__init__()
        self.hp = hyper_parameters
        self.path = self.hp.local.local_path + self.hp.local.data_path

    def get_dataset_(self, shop_name):
        meta: 'Shop' = self.hp.shops.meta[shop_name]
        start_date, end_date = self.set_date(self.hp, meta)
        df = self.load_dataset(meta.shop_id, start_date, end_date)

        if self.hp.local.update_local:
            folder_path = os.path.join(self.path, "datasets")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, f"{meta.shop_id}.csv")
            df.to_csv(file_path)
        return df

    def get_meta_(self, shop_name, f: callable, folder: str, offer_ids):
        """
        loads json from a folder :
        min_max_prices
        offer_ids
        pure_costs
        stocks
        volume
        volume_costs
        """
        meta: 'Shop' = self.hp.shops.meta[shop_name]

        print(f"enter: get_meta_ for shop_name: {shop_name} : folder: {folder}")
        metric = dict()
        for offer_id in offer_ids:
            print(offer_id)
            metric[offer_id] = f(meta.shop_id, offer_id)

        if self.hp.local.update_local:
            folder_path: str = os.path.join(self.path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, f"{meta.shop_id}.json")
            with open(file_path, "w") as f:
                json.dump(metric, f, indent=4, sort_keys=False)

        return metric

    def load_dp_offer_ids(self, shop_id):
        from dynamic_pricing.core.read_params import ReadParams
        try:
            item = list(ReadParams.read(shop_id=shop_id)["settings"].keys())
        except Exception as e:
            print(f"Unsuccessful load_dp_offer_ids for: {shop_id}")
            print(e)
            item = None

        if self.hp.local.update_local:
            folder_path = os.path.join(self.path, "dp_offer_ids")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, f"{shop_id}.json")
            with open(file_path, "w") as f:
                json.dump(item, f, indent=4, sort_keys=False)

        return item


class DatasetLoaderUtils:
    def __init__(self):
        pass

    def load_offer_ids(self, shop_id: str):
        path = f"/Users/dev.daniil.bakushkin/Desktop/suppi/backend\
/dynamic_pricing/ml/release/model/data_backup/data/offer_ids/{shop_id}.json"
        with open(path, "r") as f:
            metric = json.load(f)
        return metric


class DatasetLoaderLocal:
    def __init__(self, hyper_parameters: 'HyperParameters') -> None:
        super().__init__()
        self.hp = hyper_parameters
        self.path = self.hp.local.local_path + self.hp.local.data_path

    def get_dataset_(self, shop_name):
        meta: 'Shop' = self.hp.shops.meta[shop_name]
        path = self.path + f"datasets/{meta.shop_id}.csv"

        df = pd.read_csv(path)
        return df

    def get_meta_(self, shop_name, folder):
        """
        loads json from a folder :
        min_max_prices
        offer_ids
        pure_costs
        stocks
        volume
        volume_costs
        """
        meta: 'Shop' = self.hp.shops.meta[shop_name]
        path = self.path + f"{folder}/{meta.shop_id}.json"
        with open(path, "r") as f:
            metric = json.load(f)
        return metric


class DatasetLoader:
    def __init__(self, hyper_parameters: 'HyperParameters') -> None:
        self.hp = hyper_parameters
        self.loader = None
        self.init_loader()

    def init_loader(self):
        loader = DatasetLoaderLocal(self.hp)
        self.loader = loader

    def get_dataset(self, shop_name):
        print(f"enter: get_dataset: {shop_name}")
        return self.loader.get_dataset_(shop_name)

    def get_meta(self, shop_name, folder):
        print(f"enter: get_meta: {shop_name} folder: {folder}")
        return self.loader.get_meta_(shop_name, folder)
