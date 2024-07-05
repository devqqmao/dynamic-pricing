from datetime import date
from sklearn.preprocessing import LabelEncoder


class PredictionGoal:
    def __init__(self, sellout=None):
        if sellout is None:
            self.sellout = True
        else:
            self.sellout = sellout


class Mode:
    def __init__(self):
        self.val_mode_on: bool = True
        self.load_meta_from_server: bool = False
        self.load_dataset_from_server: bool = False

        self.train_on_dates = True

        self.train_on_meta_offers = True
        self.train_on_all_offers = False
        self.train_on_dp_offers = False


class Shop:
    def __init__(self, shop_id, offer_ids, start_date, end_date, days_on_sale=None):
        self.shop_id = shop_id
        self.offer_ids = offer_ids
        self.start_date = start_date
        self.end_date = end_date
        self.days_on_sale = days_on_sale


class Shops:
    def __init__(self):
        self.default_start_date = date(2021, 1, 1)
        self.default_end_date = date.today()
        self.delay_timeline: int = 4
        self.product_ids = []
        self.le_product_ids = LabelEncoder()

        self.shop_name_to_id = {"twist": "479580b8-30b1-4ace-9da1-77649e3c39ee",
                                "tomzn": "5032acdf-3b10-42d1-a77d-a10bab61e29c",
                                "stroimarket": "0cf90fda-735b-405b-93a9-6e04ae41d941",
                                "aktivmarket": "4681ce04-577d-48fc-8a5b-c508a871d812",
                                "kojima": "2f2b5eb3-d0cf-44e1-89f5-fc165def913c",
                                "aquacenter": "f7affe16-ce76-4789-9dd7-dfa45080fd59",
                                "simulated": "shop"}

        self.shops = ["simulated"]
        # self.shops = ["aquacenter"]
        # self.shops = ["twist"]

        # train_on_meta или фул сбор датасета с чистого
        # train_on_meta не используется нигде после сбора общего датасета

        self.meta = {"twist": Shop(
            shop_id=self.shop_name_to_id["twist"],
            offer_ids=[],
            start_date=date(2023, 9, 1),
            end_date=date(2024, 3, 1),
            days_on_sale=60,
        ),
            "tomzn": Shop(shop_id=self.shop_name_to_id["tomzn"],
                          offer_ids=[],
                          start_date=date(2024, 4, 1),
                          end_date=self.default_end_date,
                          days_on_sale=60,
                          ),
            "stroimarket": Shop(shop_id=self.shop_name_to_id["stroimarket"],
                                offer_ids=[],
                                start_date=date(2023, 9, 1),
                                end_date=self.default_end_date,
                                days_on_sale=60,
                                ),
            "kojima": Shop(shop_id=self.shop_name_to_id["kojima"],
                           offer_ids=[],
                           start_date=date(2023, 9, 1),
                           end_date=date(2024, 3, 1),
                           days_on_sale=60,
                           ),
            "aquacenter": Shop(shop_id=self.shop_name_to_id["aquacenter"],
                               # for sellout

                               # offer_ids=[
                               #     "520113",
                               #     "520021"
                               # ],

                               offer_ids=[
                                   "110011",
                                   "111004",
                                   "111012",
                                   "111015",
                                   "110009",
                                   "110010",
                                   "130000",
                                   "121107",
                                   "120100",
                                   "121353",
                               ],
                               start_date=date(2023, 9, 1),
                               end_date=date(2024, 3, 1),
                               days_on_sale=30,
                               ),
            "aktivmarket": Shop(shop_id=self.shop_name_to_id["aktivmarket"],
                                # for sellout
                                offer_ids=[],
                                start_date=date(2023, 9, 1),
                                end_date=self.default_end_date,
                                days_on_sale=60,
                                ),
            "simulated": Shop(shop_id=self.shop_name_to_id["simulated"],
                              offer_ids=["22_1",
                                         "41_1",
                                         "61_1",
                                         "20_1",
                                         "60_1",
                                         "2_1",
                                         "29_1",
                                         "30_1",
                                         "5_1",
                                         "10_1"],
                              start_date=date(2015, 1, 1),
                              end_date=date(2016, 5, 23),
                              days_on_sale=200),
        }


class TechFeatures:
    def __init__(self, days_as_features=None):
        if days_as_features is None:
            self.days_as_features: int = 14
        else:
            self.days_as_features = days_as_features
        self.n_features_stacked: int = 2
        self.prediction_period: int = 14
        self.delay_prediction: int = 1


class Local:
    def __init__(self):
        self.update_local: bool = True

        self.retrain_model = True
        self.save_model = True
        self.load_model = False

        # self.local_path = "dynamic_pricing/ml/"
        self.local_path = "/Users/dev.daniil.bakushkin/Desktop/suppi/backend/dynamic_pricing/ml/"
        self.data_path = "release/model/data/"
        self.model_meta_path = self.local_path + "release/model/fit/model_files/"
        self.catboost_model_path = self.model_meta_path + "model.cbm"
        self.catboost_feature_importances_path = self.model_meta_path + "feature_importances.pkl"


class HyperParameters:

    def __init__(self,
                 prediction_goal: 'PredictionGoal',
                 mode: 'Mode',
                 shops: 'Shops',
                 tech_features: 'TechFeatures',
                 local: 'Local',
                 ):
        self.prediction_goal = prediction_goal
        self.mode = mode
        self.shops = shops
        self.tech_features = tech_features
        self.local = local


def init_hyper_parameters(sellout=None, days_as_features=None):
    prediction_goal = PredictionGoal(sellout)
    val_mode = Mode()
    tech_features = TechFeatures(days_as_features)
    shops = Shops()
    local = Local()
    hp = HyperParameters(prediction_goal,
                         val_mode,
                         shops,
                         tech_features,
                         local)
    return hp
