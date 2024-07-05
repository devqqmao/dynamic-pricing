from typing import Union, List

from dynamic_pricing.ml.release.model.preprocess.collect_datasets import DatasetsCollector

from typing import TYPE_CHECKING

from dynamic_pricing.ml.release.model.fit.create_matrix import MatrixGenerator
from dynamic_pricing.ml.release.model.fit.model import ModelWrapper
from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder
from dynamic_pricing.ml.release.model.predict.demand.predictor import DemandPredictor, BestPricePredictor
from dynamic_pricing.ml.release.model.predict.sellout.predictor import SelloutPredictor, PricePickerSellout

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters


# класс со всеми переменными

class PipeLine:
    def __init__(self):
        pass

    def _pipeline_load_and_preprocess(self, hp: 'HyperParameters'):
        print("enter: _pipeline_load_and_preprocess")
        datasets_collector = DatasetsCollector(hp)
        offers_holder = datasets_collector.collect_datasets()
        return offers_holder

    def _pipeline_fit(self, offers_holder: 'OffersHolder'):
        print("enter: _pipeline_load_and_preprocess")
        # Создаем продуктовые матрицы

        matrix_generator = MatrixGenerator(offers_holder)
        X_train, y_train = matrix_generator.get_matrix()

        # Обучаем модель
        model_wrapper = ModelWrapper(offers_holder)
        model_catboost = model_wrapper.fit(X_train, y_train)

        model_wrapper.model_catboost = model_catboost
        offers_holder.model_wrapper = model_wrapper

        return offers_holder

    def _pipeline_predict_demand(self, holder: 'OffersHolder'):
        print("enter: _pipeline_predict_demand")
        demand_predictor = DemandPredictor(holder)
        price_predictor = BestPricePredictor(holder)

        demand_predictor.update_demand_metaholder()
        recommended_prices = price_predictor.predict()
        return holder, recommended_prices

    def _pipeline_predict_sellout(self, offers_holder: 'OffersHolder'):
        """
        Функция принимает shop_id и offer_ids и загружает вилки возможных цен по ним.

        Parameters
        ----------
        shop_id : string
            ID
        offer_ids : List[string]
             Товары

        Return
        ------
        dict : Dict[string, Tuple[float, float]]
            Dict with Lists:
            Ключи – offer_id
            Значения – [min_price, max_price]

        """
        print("enter: _pipeline_predict_sellout")
        predictor = SelloutPredictor(offers_holder)
        offers_holder, recommended_prices = predictor.predict()

        return offers_holder, recommended_prices

    def run_pipeline(self, hp: 'HyperParameters'):
        print("enter: run_pipeline")
        holder = self._pipeline_load_and_preprocess(hp)
        holder = self._pipeline_fit(holder)
        holder, recommended_prices_demand = self._pipeline_predict_demand(holder)

        recommended_prices_sellout = None
        if hp.prediction_goal.sellout:
            holder, recommended_prices_sellout = self._pipeline_predict_sellout(holder)

        return holder, recommended_prices_demand, recommended_prices_sellout
