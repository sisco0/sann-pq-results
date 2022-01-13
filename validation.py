import tensorflow as tf
import numpy as np
import os
from sannpq import estimate as sannpq
from gapsopq import estimate as gapso
from pqsignal import generate_batch as signalgenpq
import pandas as pd
import matplotlib.pyplot as plt

def validationopt(dict, method, methodidx, methodname, nc, sptc, t, signals, categories, sps, tmask, tmaskres, plot=False):
    for k in range(len(signals)):
        while True:
            print('Category: ', categories[k])
            c = k+len(signals)*methodidx
            signal = signals[k]
            result = method(t, signal, sps, 60.0, tmask, tmaskres, plot)
            if result["rsse"] < 0.1:
                break
            else:
                plt.figure()
                plt.plot(t, signal)
                plt.show()
        dict["method"][c] = methodidx
        dict["rsse"][c] = result["rsse"]
        dict["mad"][c] = result["mad"]
        dict["duration"][c] = result["duration"]
        dict["category"][c] = categories[k]
    return dict


if __name__ == '__main__':
    """Validation of the SANN-PQ optimization methodology
    by the estimation of the parameters of synthetic power signals.
    A comparison is performed versus one methodology:
    1. GA-PSO optimization methodology."""
    # Parameters for tests
    nc = [1, 2, 3, 4, 5, 6, 7, 8]  # categories
    sps = 3000  # Sampling frequency
    length = 0.2
    sptc = 10000
    # Create the synthetic signals for each test case
    t = np.arange(-0.5, length+0.5, 1.0/sps)
    tmask = np.where((t >= 0) & (t < length))
    tmaskres = np.where((t >= 1e-2) & (t < length-1e-2))
    signals = [signalgenpq(k, t, sptc) for k in
               nc]
    signals = np.reshape(signals, (len(nc)*sptc, len(t)))
    categories = np.repeat(nc, sptc)

    dict = {
        "method": np.zeros(len(nc)*sptc*2, dtype='int'),
        "category": np.zeros(len(nc)*sptc*2, dtype='int'),
        "rsse": np.zeros(len(nc)*sptc*2),
        "mad": np.zeros(len(nc)*sptc*2),
        "duration": np.zeros(len(nc)*sptc*2)
    }
    dict = validationopt(dict, sannpq, 0, "sannpq", nc,
                         sptc, t, signals, categories, sps, tmask, tmaskres)
    dict = validationopt(dict, gapso, 1, "gapso", nc, sptc,
                         t, signals, categories, sps, tmask, tmaskres)
    df = pd.DataFrame(data=dict)
    print(df)
    print(df.groupby(['method', 'category']).mean())
    print(df.groupby(['method', 'category']).std())
    df.to_csv('validation.csv')
