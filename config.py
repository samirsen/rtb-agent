import os
import _pickle as pickle
import numpy as np

data_path = "data/ipinyou-data/"
camps = ["1458", "2259", "2261", "2821", "2997",
         "3358", "3386", "3427", "3476"]

max_market_price = 300

info_keys = ["imp_test", "cost_test", "clk_test", "imp_train", "cost_train",
             "clk_train", "field", "dim", "price_counter_train"]

def get_camp_info(camp):
    return pickle.load(open(os.path.join(data_path, camp) + "/info.txt", "rb"))
