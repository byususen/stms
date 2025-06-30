#SPATIOTEMPORAL FILLING - MULTISTEP SMOOTHING DATA RECONSTRUCTION
# Author: Bayu Suseno <bayu.suseno@outlook.com>

import numpy as np
from pygam import LinearGAM, s
from tqdm.auto import tqdm
import math
import time

class stms:
    """
    Spatiotemporal Filling - Multistep Smoothing (STMS) for reconstructing satellite-derived 
    vegetation index (VI) time series data in cloud-prone regions.

    This class handles:
    - Detection of VI time series with long cloudy periods.
    - Spatiotemporal filling using nearby spatial samples with high correlation and low cloud contamination.
    - Iterative smoothing using Generalized Additive Models (GAM) to refine VI values.

    Parameters
    ----------
    n_spline : int
        Number of splines used in GAM smoothing.
    smoothing_min : float
        Minimum cloud threshold used during multistep smoothing.
    smoothing_max : float
        Maximum cloud threshold used during multistep smoothing.
    smoothing_increment : float
        Step size for increasing the cloud threshold in multistep smoothing.
    lamdas : array-like
        Regularization parameters for GAM during grid search.
    vi_max : float
        Maximum allowable value for VI predictions.
    vi_min : float
        Minimum allowable value for VI predictions.
    n_consecutive : int
        Minimum number of consecutive cloudy observations to trigger spatiotemporal filling.
    n_tail : int
        Number of extra time steps (padding) added before/after the cloudy interval.
    threshold_cloudy : float
        Threshold below which a time point is considered “clear”.
    threshold_corr : float
        Minimum correlation required between target and candidate series.
    n_candidate : int
        Maximum number of spatially similar series to use for reconstruction.
    """
    
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
        """
        Reconstructs VI values in intervals with prolonged cloudy conditions by using nearby
        samples with similar temporal patterns.

        Parameters
        ----------
        id_sample : array-like
            Sample ID for each observation.
        days_data : array-like
            Day-of-year (or timestamp) for each observation.
        vi_data : array-like
            Vegetation index (e.g., NDVI) values, possibly containing cloudy/noisy data.
        long_data : array-like
            Longitude of each sample.
        lati_data : array-like
            Latitude of each sample.
        cloud_data : array-like
            CloudScore+ or similar quality measure (lower = clearer observation).

        Returns
        -------
        vi_data : np.ndarray
            Reconstructed vegetation index array with cloud-contaminated values filled.
        """
        vi_raw = vi_data
        idsamp_unique = np.unique(id_sample)
        id_gap = np.empty(0, dtype=object)
        days_gap = np.empty(0, dtype=int)
        long_gap = np.empty(0, dtype=float)
        lati_gap = np.empty(0, dtype=float)
        vi_gap = np.empty(0, dtype=float)
        cloud_gap = np.empty(0, dtype=float)
        for i in tqdm(idsamp_unique, desc="STEP 1. Finding series with consecutive cloudy condition"):
            time.sleep(0.1)
            id_values = id_sample[id_sample==i]
            days_values = days_data[id_sample==i]
            long_values = long_data[id_sample==i]
            lati_values = lati_data[id_sample==i]
            vi_values = vi_data[id_sample==i]
            cloud_values = cloud_data[id_sample==i]
            filter_cloud = cloud_values <= self.threshold_cloudy
            cons_temp = 0
            cons_max = 0
            for j in range(len(filter_cloud)) :
                if filter_cloud[j] == True :
                    cons_temp += 1
                else :
                    cons_temp = 0
                if cons_temp > cons_max :
                    cons_max = cons_temp
            if cons_max >= self.n_consecutive:
                id_gap = np.append(id_gap,id_values)
                days_gap = np.append(days_gap,days_values)
                long_gap = np.append(long_gap,long_values)
                lati_gap = np.append(lati_gap,lati_values)
                vi_gap = np.append(vi_gap,vi_values)
                cloud_gap = np.append(cloud_gap,cloud_values)
        #create interval
        count = 1
        unique_int = np.empty(0, dtype=int)
        id_int = np.empty(0, dtype=object)
        days_int = np.empty(0, dtype=int)
        long_int = np.empty(0, dtype=float)
        lati_int = np.empty(0, dtype=float)
        vi_int = np.empty(0, dtype=float)
        cloud_int = np.empty(0, dtype=float)
        for i in tqdm(np.unique(id_gap), desc="STEP 2. Creating target interval"):
            id_values = id_gap[id_gap==i]
            days_values = days_gap[id_gap==i]
            long_values = long_gap[id_gap==i]
            lati_values = lati_gap[id_gap==i]
            vi_values = vi_gap[id_gap==i]
            cloud_values = cloud_gap[id_gap==i]
            filter_cloud = cloud_values <= self.threshold_cloudy
            cons_temp = 0
            cons_max = 0
            for j in range(len(filter_cloud)) :
                if (filter_cloud[j] == True) and (j < len(filter_cloud) - 1):
                    cons_temp += 1
                elif (filter_cloud[j]) == True and (j == len(filter_cloud) - 1):
                    cons_max = cons_temp + 1
                    cons_temp = 0
                    if cons_max >= self.n_consecutive :
                        if j - cons_max - self.n_tail < 0 :
                            id_temp = id_values[0:j + self.n_tail +1]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[0:j + self.n_tail +1]
                            long_temp = long_values[0:j + self.n_tail +1]
                            lati_temp = lati_values[0:j + self.n_tail +1]
                            vi_temp = vi_values[0:j + self.n_tail +1]
                            cloud_temp = cloud_values[0:j + self.n_tail +1]
                        elif j + self.n_tail >= len(filter_cloud):
                            id_temp = id_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            long_temp = long_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                        else:
                            id_temp = id_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                        cons_max = 0
                        id_int = np.append(id_int, id_temp)
                        unique_int = np.append(unique_int, unique_temp)
                        days_int = np.append(days_int, days_temp)
                        long_int = np.append(long_int, long_temp)
                        lati_int = np.append(lati_int, lati_temp)
                        vi_int = np.append(vi_int, vi_temp)
                        cloud_int = np.append(cloud_int, cloud_temp)
                        count += 1
                else :
                    cons_max = cons_temp
                    cons_temp = 0
                    if cons_max >= self.n_consecutive :
                        if j - cons_max - self.n_tail < 0 :
                            id_temp = id_values[0:j + self.n_tail +1]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[0:j + self.n_tail +1]
                            long_temp = long_values[0:j + self.n_tail +1]
                            lati_temp = lati_values[0:j + self.n_tail +1]
                            vi_temp = vi_values[0:j + self.n_tail +1]
                            cloud_temp = cloud_values[0:j + self.n_tail +1]
                        elif j + self.n_tail >= len(filter_cloud):
                            id_temp = id_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            long_temp = long_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1 : len(filter_cloud)]
                        else:
                            id_temp = id_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            unique_temp = np.empty(len(id_temp), dtype=int); unique_temp.fill(count)
                            days_temp = days_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                        cons_max = 0
                        id_int = np.append(id_int,id_temp)
                        unique_int = np.append(unique_int,unique_temp)
                        days_int = np.append(days_int,days_temp)
                        long_int = np.append(long_int,long_temp)
                        lati_int = np.append(lati_int,lati_temp)
                        vi_int = np.append(vi_int,vi_temp)
                        cloud_int = np.append(cloud_int,cloud_temp)
                        count += 1    
        #Spatiotemporal filling
        for i in tqdm(np.unique(unique_int), desc="STEP 3. Spatiotemporal filling"): 
            id_target = id_int[unique_int==i]
            days_target = days_int[unique_int==i]
            long_target = long_int[unique_int==i]
            lati_target = lati_int[unique_int==i]
            vi_target = vi_int[unique_int==i]
            cloud_target = cloud_int[unique_int==i]
            vi_cand_pred_all = np.empty(0, dtype=float)
            weight_cand_pred_all = np.empty(0, dtype=float)
            distance_cand_all = np.empty(0, dtype=float)
            corr_cand_all = np.empty(0, dtype=float)
            vi_cand_all = np.empty(0, dtype=float)
            filter_cand_all = np.empty(0, dtype=int)
            filterpred_cand_all = np.empty(0, dtype=int)
            distance_values = np.empty(len(idsamp_unique), dtype=float); distance_values.fill(0.0000001)
            loc_target = np.array((lati_target[0], long_target[0]))
            filter_target_lo = cloud_target <= self.threshold_cloudy
            filter_target_hi = cloud_target > self.threshold_cloudy
            for j in idsamp_unique :
                lati_cand = lati_data[id_sample == j]
                long_cand = long_data[id_sample == j]
                loc_cand = np.array((lati_cand[0], long_cand[0]))
                distance_temp = math.dist(loc_target,loc_cand)
                if distance_temp != 0:
                    distance_values[idsamp_unique==j] = distance_temp
            distance_invers = np.reciprocal(distance_values)
            distance_norm = np.abs((distance_invers-np.min(distance_invers))/(np.max(distance_invers)-np.min(distance_invers)))
            distance_norm[distance_norm <= 0] = 0.0000001
            index_distance = np.argsort(distance_norm)
            idsamp_sort = idsamp_unique[index_distance]
            for k in idsamp_sort :
                id_cand = id_sample[id_sample==k]
                days_cand = days_data[id_sample==k]
                long_cand = long_data[id_sample==k]
                lati_cand = lati_data[id_sample==k]
                vi_cand = vi_raw[id_sample==k]
                cloud_cand = cloud_data[id_sample==k]
                distance_cand = distance_norm[idsamp_unique==k]
                last_row = len(vi_cand)
                filter_cand_hi = cloud_cand > self.threshold_cloudy
                first_row = 0
                end_row = len(vi_target)
                corr_temp = 0
                while end_row <= last_row :
                    filter_hi = filter_target_hi * filter_cand_hi[first_row : end_row]
                    filter_pred = filter_target_lo * filter_cand_hi[first_row : end_row]
                    if np.sum(filter_hi) >= np.sum(filter_target_hi)/2 and np.sum(filter_pred) > 0 :
                        vi_cand_cut = vi_cand[first_row : end_row]
                        cloud_cand_cut = cloud_cand[first_row : end_row]
                        corr_value = np.corrcoef(vi_target[filter_hi], vi_cand_cut[filter_hi])[0, 1]
                        if corr_value > corr_temp :
                            corr_temp = corr_value
                            vi_cand_temp = vi_cand_cut
                            cloud_cand_temp = cloud_cand_cut
                            filter_temp = filter_hi
                            filter_pred_temp = filter_pred
                    if corr_temp < 0.3 or corr_value < 0.3:
                        first_row += 3
                        end_row += 3
                    else :
                        first_row += 1
                        end_row += 1
                if corr_temp >= self.threshold_corr :
                    distance_cand_all = np.append(distance_cand_all,distance_cand)
                    corr_cand_all = np.append(corr_cand_all,corr_temp)
                    vi_cand_all = np.append(vi_cand_all,vi_cand_temp)
                    filter_cand_all = np.append(filter_cand_all,filter_temp)
                    filterpred_cand_all = np.append(filterpred_cand_all,filter_pred_temp)
                if len(corr_cand_all) == self.n_candidate:
                    break
            for l in range(len(corr_cand_all)) :
                distance_cand_temp = distance_cand_all[l]
                corr_cand_temp = corr_cand_all[l]
                vi_cand_temp = vi_cand_all[l*len(vi_target):(l+1)*len(vi_target)]
                filter_cand_temp = filter_cand_all[l*len(vi_target):(l+1)*len(vi_target)]
                filterpred_cand_temp = filterpred_cand_all[l*len(vi_target):(l+1)*len(vi_target)]
                model_coef = np.polyfit(x=vi_cand_temp[filter_cand_temp == 1], y=vi_target[filter_cand_temp == 1], deg=3)
                model_pred = np.poly1d(model_coef)
                vi_cand_pred = model_pred(vi_cand_temp)
                vi_cand_pred[vi_cand_pred > self.vi_max] = self.vi_max
                vi_cand_pred[vi_cand_pred < self.vi_min] = self.vi_min
                vi_cand_pred[filterpred_cand_temp == 0] = 0
                vi_cand_pred_fin = vi_cand_pred * corr_cand_temp * distance_cand_temp
                weight_cand_pred = filterpred_cand_temp * corr_cand_temp * distance_cand_temp
                weight_cand_pred[filterpred_cand_temp == 0] = 0
                vi_cand_pred_all = np.append(vi_cand_pred_all,vi_cand_pred_fin)
                weight_cand_pred_all = np.append(weight_cand_pred_all,weight_cand_pred)
                if len(vi_cand_pred_all) > 0 :
                    vi_pred_sum = np.nansum(np.split(vi_cand_pred_all,int(len(vi_cand_pred_all)/len(vi_target))), axis = 0)
                    filterpred_sum = np.nansum(np.split(filterpred_cand_all,int(len(filterpred_cand_all)/len(vi_target))), axis = 0)
                    weight_pred_sum = np.nansum(np.split(weight_cand_pred_all,int(len(weight_cand_pred_all)/len(vi_target))), axis = 0)
                    vi_pred_fin = vi_pred_sum / weight_pred_sum
                    vi_pred_fin[np.abs(vi_pred_fin) > 1] = vi_target[np.abs(vi_pred_fin) > 1]
                    for m in days_target[filterpred_sum > 0]:
                        vi_data[np.where(np.logical_and( id_sample == id_target[0], days_data == m))[0]] = vi_pred_fin[days_target == m]
        return vi_data

    def multistep_smoothing(self, id_sample, days_data, vi_data, cloud_data):
        """
        Applies iterative GAM-based smoothing on each time series using progressively stricter
        cloud confidence weights.

        Parameters
        ----------
        id_sample : array-like
            Sample ID for each observation.
        days_data : array-like
            Day-of-year or temporal axis.
        vi_data : array-like
            Vegetation index values to be smoothed.
        cloud_data : array-like
            Initial cloud confidence weights (e.g., CloudScore+).

        Returns
        -------
        vi_data : np.ndarray
            Smoothed vegetation index values after multistep GAM filtering.
        """
        idsamp_unique = np.unique(id_sample)
        cloud_init = cloud_data.copy() 
        for i in tqdm(idsamp_unique, desc="Multistep Smoothing"):
            id_values = id_sample[id_sample==i]
            days_values = days_data[id_sample==i]
            vi_values = vi_data[id_sample==i]
            cloud_init_sample = cloud_init[id_sample==i]
            cloud_values = cloud_data[id_sample==i]
            for j in np.arange(self.smoothing_min,self.smoothing_max + self.smoothing_increment,self.smoothing_increment):
                gam = LinearGAM(s(0),n_splines=self.n_spline).fit(days_values, vi_values, weights = cloud_values)
                gam.gridsearch(days_values, vi_values, weights = cloud_values, lam=self.lamdas, objective='GCV', progress=False)
                vi_smooth = gam.predict(days_values)
                vi_diff = vi_smooth - vi_values
                vi_smooth[vi_smooth > 1] = 1
                vi_smooth[vi_smooth < -1] = -1
                cloud_new = np.abs((vi_diff-np.min(vi_diff))/(np.max(vi_diff)-np.min(vi_diff)))
                cloud_new[cloud_new == 0] = 0.0000001
                cloud_new[(cloud_init_sample > j) & (cloud_new <= cloud_init_sample)] = cloud_init_sample[(cloud_init_sample > j) & (cloud_new <= cloud_init_sample)]
                cloud_new[(cloud_init_sample <= j) & (cloud_new > cloud_init_sample)] = cloud_init_sample[(cloud_init_sample <= j) & (cloud_new > cloud_init_sample)]
                vi_values[cloud_init_sample <= j] = vi_smooth[cloud_init_sample <= j]
                cloud_values = cloud_new
            vi_data[id_sample==i] = vi_values
        return vi_data
