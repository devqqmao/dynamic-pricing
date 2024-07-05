from typing import Union, List

from dynamic_pricing.ml.release.utils.inference import *


def define_hyper_params():
    """
    Единственное назначение функции – определить глобальные гиперпараметры модели.

    Parameters
    ----------
    Return
    ------
    tuple[params]
        За подробным описанием фичей нужно обратиться к декларации класса Holder
    """

    K = 14  # number of features as input
    rolling_period = K * 2  # rolling_period
    n_features_stacked = 2  # n_features_stacked
    n_features_to_update = 0  # n_features_to_update

    test_period = 0  # test_period
    val_period = 7  # val_period

    return K, rolling_period, n_features_stacked, n_features_to_update, test_period, val_period


def pipeline_load(product_ids: Union[None, List]):
    """
    Первый этап [pipeline_load, pipeline_fit, pipeline_predict], которые объединяют все функции
    для создания предсказаний.

    Parameters
    ----------
    product_ids: Union[None, List]
        Если None, то обучение будет выполнено товаров из get_specific_product_ids
        Если List, то для конкретных товаров

    Return
    ------
    holder : Class
        Класс, в котором хранятся данные, ассоциированные с каждым временным рядом.
        Подробное описание читать в месте декларации класса.
    """
    # получаем гиперпараметры
    K, rolling_period, n_features_stacked, n_features_to_update, test_period, val_period = define_hyper_params()

    # инициализируем инстанс холдер
    holder = Holder(product_ids, test_period, val_period, K,
                    n_features_stacked, n_features_to_update)
    # инитим холдер
    holder.init()

    return holder


def pipeline_fit(holder):
    """
    Второй этап [pipeline_load, pipeline_fit, pipeline_predict], которые объединяют все функции
    для создания предсказаний.

    Parameters
    ----------
    holder : Class
        Описание в pipeline_load

    Return
    ------
    holder : Class
        Внутри holder лежит обученная модель.
    """

    # Создаем продуктовые матрицы
    X_train_final_united, y_train_demand_united = create_products_matrix(
        holder)

    # Добавляем фичи продуктов
    X_train_final_united_marked = add_features_product(
        X_train_final_united, holder)
    y_train_final_united_marked = y_train_demand_united

    # Обучаем модель
    model_united = train_model(
        X_train_final_united_marked, y_train_final_united_marked)

    # Кладем модель в класс
    holder.model = model_united
    return holder


def predict_multiple_prices(holder, product_id, bounds, plot=None):
    """
    Функция – пустышка, созданная для извлечения аргументов из holder
    [Переписать]

    Parameters
    ----------
    holder : Class
        Описание в pipeline_load
    product_id : List[string]
        Описание в holder
    bounds : List[float, float]
        Границы цен для предсказания
    plot : bool
        Параметр не используется, артефакт исследований

    Return
    ------
    prices_dict: dict[float, pd.DataFrame]
        Возвращает Dict:
        Ключи – изменение цены;
        Значения – датафреймы с наборами подсчитанных метрик для изменения цены.
    """

    # извлекаем фичи
    X_train, X_val, prices, n_features_to_update, model, K, N, n_features_stacked, N_products = extract_features_from_holder(
        holder, product_id)

    # передаем в функцию
    prices_dict = calculate_prices_dict(X_train, X_val, predict, model, K, n_features_stacked,
                                        n_features_to_update, plot=plot, borders=bounds, holder=holder,
                                        product_id=product_id)

    return prices_dict


def pipeline_predict(holder, product_id_str, prices, plot):
    """
    Третий этап [pipeline_load, pipeline_fit, pipeline_predict], которые объединяют все функции
    для создания предсказаний.

    Parameters
    ----------
    holder : Class
        Описание в pipeline_load
    product_id : List[string]
        Описание в holder
    bounds : List[float, float]
        Границы цен для предсказания
    plot : bool
        Параметр не используется, артефакт исследований

    Return
    ------
    holder : Class
        Внутри holder лежит обученная модель.
    """

    # получаем уникальный product_id
    product_id = holder.product_str_to_id[product_id_str]
    # получаем границы цен для продукта
    bounds = prices[product_id_str]
    # получаем предсказания
    prices_dict = predict_multiple_prices(holder, product_id, bounds, plot)

    return prices_dict
