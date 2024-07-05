from dynamic_pricing.ml.release.model.eval.grid_search import GridSearcher
from dynamic_pricing.ml.release.model.eval.evaluate import Evaluator
from dynamic_pricing.ml.release.model.pipeline import PipeLine
from dynamic_pricing.ml.release import hyper_parameters
import utils
from dynamic_pricing.ml.release.run_model import main


def debug():
    hp_instance = hyper_parameters.init_hyper_parameters()
    pipeline = PipeLine()
    products_holder, recommended_prices_demand, recommended_prices_sellout = pipeline.run_pipeline(hp_instance)
    # eval = Evaluator(products_holder)
    # eval.evaluate()
    # gs = GridSearcher(products_holder)
    # gs.grid_search()
    # gs.get_results()
    # utils.log_results(products_holder)

    print(recommended_prices_demand)
    print(recommended_prices_sellout)

    return products_holder, recommended_prices_demand, recommended_prices_sellout


def get_dict_of_prices_for_the_next_day():
    holder, recommended_prices_demand, recommended_prices_sellout = main()
    return recommended_prices_demand


if __name__ == "__main__":
    print(get_dict_of_prices_for_the_next_day())
