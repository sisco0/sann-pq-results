import numpy as np
import scipy.fftpack
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import tensorflow.keras.backend as BE
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.benchmarks.benchmark import Benchmark
from NiaPy.task.task import StoppingTask, OptimizationType
import utils

# STRIPPED
