import pandas as pd


class PricePicker:
    """
    Данный класс используется для выбора наилучшей цены из prices_dict на основе выбранной стратегии:
    Возможные стратегии:
    1. maximize_profit
        Стратегия максимизации прибыли

    Parameters для __init__
    ----------

    Атрибуты класса
    ------

    Методы класса
    ------
    def _loop(d: pd.DataFrame, f: callable)
        Пробегается по всем ценам и выбирает лучшую на основании f

        Parameters
        ----------
        d: pd.DataFrame
            prices_dict

        f: callable
            функция, по которой брать argmax будет браться argmax

        Return
        ------
        best_price: float
            Наилучшая цена

    """

    def __init__(self) -> None:
        pass

    def _loop(self, d: pd.DataFrame, f: callable):
        "Функция для подсчета определенной метрики и выбора наилучшей по критерию f: callable среди всех цен"

        best_price = -float('inf')
        best_metric = -float('inf')

        for price, d in d.items():
            metric = f(d)
            if metric > best_metric:
                best_metric = metric
                best_price = price
        return best_price

    def maximize_profit(self, d):
        "Стратегия увеличения прибыли"

        def f(x): return (x["Profit"]).sum()

        return self._loop(d, f)
