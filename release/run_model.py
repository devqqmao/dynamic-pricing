from dynamic_pricing.ml.release.model.pipeline import PipeLine
from dynamic_pricing.ml.release import hyper_parameters
from dynamic_pricing.ml.release.model.eval.evaluate import Evaluator
from dynamic_pricing.ml.release.model.eval.grid_search import GridSearcher


def main():
    hp_instance = hyper_parameters.init_hyper_parameters()

    pipeline = PipeLine()

    holder, recommended_prices_demand, recommended_prices_sellout = pipeline.run_pipeline(hp_instance)

    return holder, recommended_prices_demand, recommended_prices_sellout


def get_dict_of_prices_for_the_next_day():
    holder, recommended_prices_demand, recommended_prices_sellout = main()
    return recommended_prices_demand


def get_price(shop_id, offer_id):
    res = get_dict_of_prices_for_the_next_day()
    shop = res[shop_id]
    price = shop.get(offer_id)
    if price is None:
        from marketplace.models import Product
        print(f"price is not generated for {shop_id}:{offer_id}")
        price = Product.objects.get(shop_id=shop_id, offer_id=offer_id).marketing_seller_price
    return price
