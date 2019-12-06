import os, sys
import numpy as np
import pandas as pd


def compute_mean_pdf(mean_counter, laplace=1):
    N = len(mean_counter)
    mean_pdf = [0] * N
    total = sum(mean_counter)
    normalized_mean = lambda count: (count + laplace) / (total + N*laplace)
    return [normalized_mean(count) for count in mean_counter]


def compute_pctr_pdf(train_data, pctr, bins):
    N, M = train_data.shape
    pctr_pdf = np.asarray([0 for i in range(0,bins)], dtype= float)
    for i in range(N):
        theta = train_data.iloc[i:i + 1, 2].values[0]
        index = np.where(pctr <= theta)[-1][-1]
        pctr_pdf[index] += 1

    return pctr_pdf / N
