import numpy as np
from typing import List
from dynamic_pricing.ml.release.utils.pipeline import pipeline_load, pipeline_fit, pipeline_predict
from dynamic_pricing.ml.release.utils.price_picker import PricePicker
from dynamic_pricing.models import DynamicPricing


def get_bundles(shop_id, offer_ids):
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
        Ключи – offer_ids
        Значения – [min_price, max_price]
    """

    from dynamic_pricing.models import DynamicPricing
    def get_min_max_price_by_offer_id(shop_id: str, offer_id: str):
        item = DynamicPricing.objects.filter(
            shop_id=shop_id, offer_id=offer_id).values("min_price", "max_price")[0]
        return item

    prices = dict()
    for offer_id in offer_ids:
        prices[offer_id] = get_min_max_price_by_offer_id(
            shop_id=shop_id, offer_id=offer_id)
    return prices

def get_recommened_prices(holder, product_ids: List[str], shop_id=None):
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

    # вилки цен
    prices = get_bundles(shop_id=shop_id, offer_ids=product_ids)
    # класс выбора цена
    price_picker = PricePicker()
    # рекомендованные цены
    recommened_prices = dict()

    # для каждого продукта
    for product_id_str in product_ids:
        # предсказания pd.DataFrame для всех цен
        prices_dict = pipeline_predict(
            holder, product_id_str, prices, plot=False)

        # выбираем оптимальную цену
        recommened_price = price_picker.maximize_profit(prices_dict)
        # делаем трим
        a_min, a_max = prices[product_id_str]["min_price"], prices[product_id_str]["max_price"]
        trimmed_price = np.clip(recommened_price, a_min=a_min, a_max=a_max)
        # сохраняем в дикт
        recommened_prices[product_id_str] = trimmed_price

    return recommened_prices


def get_price_prediction_for_tomorrow():
    """
    Функция получает предсказания оптимальных цен для максимизации прибыли для магазина "Twist" по offer_ids:
    shop_id = "479580b8-30b1-4ace-9da1-77649e3c39ee"
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
    Для функции необходимо соблюдение следующих инвариантов:
    1. Модель должна предсказывать, когда загрузились данные за предыдущий день.
    2. Горизон предсказания модели – гиперпараметр (в данной конфигурации 7 дней)
    3. Перед запуском функции (если необходимо) можно перенастроить гиперпараметры в функции define_hyper_params

    Parameters
    ----------

    Return
    ------
    df: [string, float]
        Рекомендованные цены для каждого товара
    """

    # определить магазин и товары
    shop_id = "479580b8-30b1-4ace-9da1-77649e3c39ee"
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


    # загрузить и обработать датасет
    holder = pipeline_load(product_ids=None)
    # обучить модель
    holder = pipeline_fit(holder)
    # предсказать цены
    recommened_prices = get_recommened_prices(holder, offer_ids, shop_id)

    return recommened_prices
