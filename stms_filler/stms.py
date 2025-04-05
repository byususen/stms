#SPATIOTEMPORAL FILLING - MULTISTEP SMOOTHING DATA RECONSTRUCTION
# Author: Bayu Suseno <bayu.suseno@outlook.com>

import numpy as np
from pygam import LinearGAM, s
from tqdm.auto import tqdm
import math
import time

class stms:
    def __init__(self, n_spline=20, smoothing_min = 0.1, smoothing_max = 1, smoothing_increment = 0.1, lamdas = np.logspace(-3, 2, 50), vi_max = 1, vi_min = -1, n_consecutive = 5, n_tail = 24, threshold_cloudy = 0.1, threshold_corr = 0.9, n_candidate = 10):
        self.n_spline = n_spline
        self.smoothing_min = smoothing_min
        self.smoothing_max = smoothing_max
        self.smoothing_increment = smoothing_increment
        self.lamdas = lamdas
        self.vi_max = vi_max
        self.vi_min = vi_min
        self.n_consecutive = n_consecutive
        self.n_tail = n_tail
        self.threshold_cloudy = threshold_cloudy
        self.threshold_corr = threshold_corr
        self.n_candidate = n_candidate

    def spatiotemporal_filling(self, id_sample, days_data, vi_data, long_data, lati_data, cloud_data):
        # [TRUNCATED for brevity]
        pass

    def multistep_smoothing(self, id_sample, days_data, vi_data, cloud_data):
        # [TRUNCATED for brevity]
        pass
