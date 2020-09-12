import numpy as np
import sys
import os
from scipy import interpolate
import matplotlib.pyplot as plt
import copy
from rockpool.timeseries import TSContinuous

def filter_1d(data, alpha = 0.9):
    last = data[0]
    out = np.zeros((len(data),))
    out[0] = last
    for i in range(1,len(data)):
        out[i] = alpha*out[i-1] + (1-alpha)*data[i]
        last = data[i]
    return out

def k_step_function(total_num_iter, step_size, start_k):
    stop_k = step_size
    num_reductions = int((start_k - stop_k) / step_size) + 1
    reduce_after = int(total_num_iter / num_reductions)
    reduction_indices = [i for i in range(1,total_num_iter) if (i % reduce_after) == 0]
    k_of_t = np.zeros(total_num_iter)
    if(total_num_iter > 0):
        k_of_t[0] = start_k
        for t in range(1,total_num_iter):
            if(t in reduction_indices):
                k_of_t[t] = k_of_t[t-1]-step_size
            else:
                k_of_t[t] = k_of_t[t-1]

    return k_of_t
