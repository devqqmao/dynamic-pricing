from dynamic_pricing.ml.release.model.load.load_dataset import DatasetLoader, DatasetLoaderServer, \
    DatasetLoaderServerUtils, DatasetLoaderLocal
from dynamic_pricing.ml.release.model.preprocess.preprocess_dataset import DatasetPreprocessor
from dynamic_pricing.ml.release.hyper_parameters import HyperParameters
from dynamic_pricing.ml.release.model.preprocess.extract_features import FeatureExtractor
from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop


class DatasetsCollector:
    def __init__(self, hyper_parameters: 'HyperParameters'):
        self.hp = hyper_parameters
        self.dataset_loader_local = DatasetLoader(self.hp)
        self.dataset_loader_server = DatasetLoaderServer(self.hp)
        self.dataset_preprocessor = DatasetPreprocessor(self.hp, self.dataset_loader_local)
        self.feature_extractor = FeatureExtractor(self.hp)
        self.holder = OffersHolder(self.hp, self.feature_extractor, self.dataset_loader_local)

        self.dataset_loader_local.init_loader()
        self.dataset_preprocessor.init_preprocessor()

    def _load_all_meta(self, shop_name):
        print(f"enter: _load_all_meta: {shop_name}")
        loader = DatasetLoaderServer(self.hp)

        meta: 'Shop' = self.holder.hp.shops.meta[shop_name]

        if self.hp.mode.train_on_dp_offers:
            offer_ids = loader.load_dp_offer_ids(self.hp.shops.shop_name_to_id[shop_name])
        else:
            offer_ids = self.hp.shops.meta[shop_name].offer_ids

        f = loader.load_min_max_prices
        folder = "min_max_prices"
        loader.get_meta_(shop_name, f, folder, offer_ids)

        f = loader.load_pure_costs
        folder = "pure_costs"
        loader.get_meta_(shop_name, f, folder, offer_ids)

        if self.hp.prediction_goal.sellout:
            f = loader.load_stocks
            folder = "stocks"
            loader.get_meta_(shop_name, f, folder, offer_ids)

            f = loader.load_volume
            folder = "volume"
            loader.get_meta_(shop_name, f, folder, offer_ids)

            f = loader.load_volume_cost
            folder = "volume_costs"
            loader.get_meta_(shop_name, f, folder, offer_ids)

    def collect_datasets(self):
        for shop_name in self.hp.shops.shops:



            print(f"enter: collect_datasets: {shop_name}")
            if not self.hp.mode.load_dataset_from_server:
                df = self.dataset_loader_local.get_dataset(shop_name)
            else:
                df = self.dataset_loader_server.get_dataset_(shop_name)

            # offer_ids are set by now
            if self.hp.mode.load_meta_from_server:
                self._load_all_meta(shop_name)

            df = self.dataset_preprocessor.preprocess_dataset(df, shop_name)

            self.holder.extract_and_preprocess(df, shop_name)


        self.holder.hp.shops.le_product_ids.fit(self.holder.hp.shops.product_ids)
        return self.holder
