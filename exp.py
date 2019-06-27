import tensorflow as tf
from pandas import DataFrame
import numpy as np
import json

from evaluater import Evaluater
from base import NetworkUnit
from sampler import Sampler

NN_POOL = []

# Setting
TEST_MAX_EPOCH = 100
NN_NAME_POOL = [""]
FILE_SUFFIX = ""
CONFIG_PATH = ""
CONFIG_RAND_VECTOR = []
RESULT_FILE = ""

def load_nn_pool():
    """Load all network structures from files"""
    for nn_name in NN_NAME_POOL:
        nn_path = nn_name + FILE_SUFFIX
        nn = NetworkUnit()
        nn.load_from(nn_path)
        NN_POOL.append(nn)
    return

def load_rand_vector():
    """Set all nodes randomly"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    for key, type_dic in config.items():
        if (key == "pros"):
            continue
        for attr in type_dic.values():
            attr_num = len(attr["val"])
            attr_rand_vec = [1 / attr_num for i in range(attr_num - 1)]
            CONFIG_RAND_VECTOR.extend(attr_rand_vec)

    type_num = len(config)) - 1
    type_rand_vec = [1 / type_num for i in range(type_num - 1)]
    CONFIG_RAND_VECTOR.extend(type_rand_vec)

def random_test(nn=NetworkUnit()):
    """Fix a network structure, give a setting randomly and get its score"""
    spl = Sampler()
    eva = Evaluater()
    spl.renewp(CONFIG_RAND_VECTOR)
    scores = []

    for i in range(TEST_MAX_EPOCH):
        nn.set_cell(spl.sample(len(nn.graph_part)))
        score = eva.evaluate(nn)
        scores.append(score)
    return scores

if __name__ == '__main__':
    load_nn_pool()
    load_rand_vector()
    result_dict = {}
    for nn in NN_POOL:
        nn.scores = random_test(nn)
        result_dict[nn.name] = nn.scores
    result_df = DataFrame(result_dict)
    result_df.to_csv(RESULT_FILE)

