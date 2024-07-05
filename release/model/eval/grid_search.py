import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING

from dynamic_pricing.ml.release.model.pipeline import PipeLine
from dynamic_pricing.ml.release import hyper_parameters
from dynamic_pricing.ml.release.model.eval.evaluate import Evaluator
from dynamic_pricing.ml.release.hyper_parameters import HyperParameters

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder, MetaHolder


class GridSearcher:
    def __init__(self, holder: 'OffersHolder'):
        self.holder = holder
        self.results = defaultdict(dict)

    def _run_pipeline(self, parameters: dict):
        hp_instance = hyper_parameters.init_hyper_parameters(sellout=False, **parameters)
        pipeline = PipeLine()
        products_holder, recommended_prices_demand, recommended_prices_sellout  = pipeline.run_pipeline(hp_instance)
        eval = Evaluator(products_holder)
        out = eval.evaluate()
        return out

    def grid_search(self):
        parameters = {"days_as_features": [3, 14, 21]}
        # parameters = {"days_as_features": [1, 3, 7, 14, 21, 28, 35]}
        # parameters = {"days_as_features": [7]}
        for k, v in parameters.items():
            for i in parameters[k]:
                self.results[k][i] = self._run_pipeline({k: i})

    def get_results(self):
        for k, d in self.results.items():
            print(f"for metric: {k}")
            for k, v in d.items():
                print(f"\t{k}: {v}")
