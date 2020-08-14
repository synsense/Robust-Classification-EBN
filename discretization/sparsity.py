import numpy as np
import json
from rockpool.networks import NetworkADS
import os

def discretize(W, base_weight):
    tmp = np.round(W / base_weight)
    return base_weight*tmp

def get_sparsity(M):
    return np.sum(np.asarray(M != 0, dtype=int)) / (M.shape[0]*M.shape[1])

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

base_path = "/home/julian/Documents/RobustClassificationWithEBNs/discretization/"

path_model_2bit = os.path.join(base_path, "Resources/hey_snips4.json")
path_model_3bit = os.path.join(base_path, "Resources/hey_snips8.json")
path_model_4bit = os.path.join(base_path, "Resources/hey_snips16.json")
path_model_full = os.path.join(base_path, "../figure2/Resources/hey_snips.json")

net_2bit = NetworkADS.load(path_model_2bit)
net_3bit = NetworkADS.load(path_model_3bit)
net_4bit = NetworkADS.load(path_model_4bit)
net_full = NetworkADS.load(path_model_full)

weights_2bit = net_2bit.lyrRes.weights_slow
weights_3bit = net_3bit.lyrRes.weights_slow
weights_4bit = net_4bit.lyrRes.weights_slow
weights_full = net_full.lyrRes.weights_slow

base_weight_2bit = (np.max(weights_2bit)-np.min(weights_2bit))/(3)
base_weight_3bit = (np.max(weights_3bit)-np.min(weights_3bit))/(7)
base_weight_4bit = (np.max(weights_4bit)-np.min(weights_4bit))/(15)

weights_2bit = discretize(weights_2bit, base_weight_2bit)
weights_3bit = discretize(weights_3bit, base_weight_3bit)
weights_4bit = discretize(weights_4bit, base_weight_4bit)

print("Sparsity: 2Bit", get_sparsity(weights_2bit), "3Bit", get_sparsity(weights_3bit), "4Bit", get_sparsity(weights_4bit), "Full", get_sparsity(weights_full))