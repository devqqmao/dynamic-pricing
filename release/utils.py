import pandas as pd

from dynamic_pricing.ml.release.model.preprocess.create_holders import OffersHolder


def get_offer_ids_local(shop_id):
    path = f"/Users/dev.daniil.bakushkin/Desktop/suppi/backend/dynamic_pricing/ml/release/model/debug/data_real/datasets/dataset_{shop_id}.csv"
    df: pd.DataFrame = pd.read_csv(path)
    return list(df['offer_id'].unique())

def get_dp_offer_ids_server(shop_id):
    path = f"/Users/dev.daniil.bakushkin/Desktop/suppi/backend/dynamic_pricing/ml/release/model/debug/data_real/datasets/dataset_{shop_id}.csv"
    df: pd.DataFrame = pd.read_csv(path)
    return list(df['offer_id'].unique())


def log_df(df: pd.DataFrame):
    print("log_df")
    print("-" * 10)
    print(df.shape)
    print(df.describe())
    print(df.head())
    print("-" * 10)


def log_results(products_holder):
    products_holder: 'OffersHolder'
    for product_id in products_holder.hp.shops.product_ids:
        data = products_holder.product_storage[product_id]
        avg_price = data.avg_price

        print(product_id)
        print(data.min_max_price)
        print(avg_price + -2 * data.price_std, avg_price, avg_price + 2 * data.price_std)
        print("demand recommended_price")
        print(products_holder.demand_metrics[product_id].recommended_price)
        print("sellout recommended_price")
        print(products_holder.sellout_metrics[product_id].recommended_price)

        # print(products_holder.demand_metrics[product_id].price_storage)
        # print(products_holder.sellout_metrics[product_id].price_storage)
