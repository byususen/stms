# SPATIOTEMPORAL FILLING - MULTISTEP SMOOTHING DATA RECONSTRUCTION
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
    vi_max : float or None
        Optional maximum allowable value for VI predictions.
        If None (default), no upper clipping is applied.
    vi_min : float or None
        Optional minimum allowable value for VI predictions.
        If None (default), no lower clipping is applied.
    n_consecutive : int
        Minimum number of consecutive cloudy observations to trigger spatiotemporal filling.
    n_tail : int
        Number of extra time steps (padding) added before/after the cloudy interval.
    threshold_cloudy : float
        Threshold below which a time point is considered “clear”.
    threshold_corr : float
        Minimum correlation required between target and candidate series.
    n_candidate : int or None
        Maximum number of spatially similar series to use for reconstruction (global cap).
        If None, no explicit global limit is applied (all accepted candidates in the pool are used).
    n_candidate_nested : int or None
        Maximum number of accepted candidates per nested group (e.g., per ID_SEGMEN).
        Only used if id_nested is provided. If None, per-nested candidates are unlimited.
    candidate_sampling : {"distance", "random"}
        Strategy to order candidate samples when searching for donors.
    max_candidate_pool : int or None
        Optional limit on how many candidate series are *tested* for each interval.
        If None, all available series are considered.
    step_min : int
        Minimum shift of the moving window when correlation is high (fine search).
    step_max : int
        Maximum shift of the moving window when correlation is low (coarse search).
    """

    def __init__(
        self,
        n_spline=20,
        smoothing_min=0.1,
        smoothing_max=1,
        smoothing_increment=0.1,
        lamdas=np.logspace(-3, 2, 50),
        vi_max=None,             
        vi_min=None,             
        n_consecutive=5,
        n_tail=24,
        threshold_cloudy=0.1,
        threshold_corr=0.9,
        n_candidate=None,            
        n_candidate_nested=None,     
        candidate_sampling="distance",
        max_candidate_pool=None,     
        step_min=1,                  
        step_max=6,                  
    ):
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
        self.n_candidate_nested = n_candidate_nested
        self.candidate_sampling = candidate_sampling
        self.max_candidate_pool = max_candidate_pool
        self.step_min = step_min
        self.step_max = step_max

    # ------------------------------------------------------------------
    # spatiotemporal_filling
    # ------------------------------------------------------------------
    def spatiotemporal_filling(
        self,
        id_sample,
        days_data,
        vi_data,
        long_data,
        lati_data,
        cloud_data,
        id_nested=None,
        candidate_sampling=None,
        max_candidate_pool=None,
    ):
        """
        Reconstructs VI values in intervals with prolonged cloudy conditions by using nearby
        samples with similar temporal patterns.

        Parameters
        ----------
        id_sample : array-like
            Sample ID for each observation (e.g., ID_SUBSEGMEN).
        ...
        vi_data : np.ndarray
            Reconstructed vegetation index array with cloud-contaminated values filled.
        """
        vi_raw = vi_data.copy()
        idsamp_unique = np.unique(id_sample)

        if candidate_sampling is None:
            candidate_sampling = self.candidate_sampling
        if max_candidate_pool is None:
            max_candidate_pool = self.max_candidate_pool

        # Optional mapping from id_sample -> nested ID (e.g. SUBSEGMEN -> SEGMEN)
        nested_by_id = None
        if id_nested is not None:
            nested_by_id = {}
            for sid in idsamp_unique:
                nested_by_id[sid] = id_nested[id_sample == sid][0]

        # Precompute one lat/long per id_sample to avoid repeated indexing
        lat_by_id = {}
        lon_by_id = {}
        for sid in idsamp_unique:
            mask = (id_sample == sid)
            lat_by_id[sid] = lati_data[mask][0]
            lon_by_id[sid] = long_data[mask][0]

        # STEP 1. Finding series with consecutive cloudy condition
        id_gap = np.empty(0, dtype=object)
        days_gap = np.empty(0, dtype=int)
        long_gap = np.empty(0, dtype=float)
        lati_gap = np.empty(0, dtype=float)
        vi_gap = np.empty(0, dtype=float)
        cloud_gap = np.empty(0, dtype=float)

        for i in tqdm(idsamp_unique, desc="STEP 1. Finding series with consecutive cloudy condition"):
            time.sleep(0.1)
            mask = (id_sample == i)
            id_values = id_sample[mask]
            days_values = days_data[mask]
            long_values = long_data[mask]
            lati_values = lati_data[mask]
            vi_values = vi_data[mask]
            cloud_values = cloud_data[mask]

            filter_cloud = (cloud_values <= self.threshold_cloudy)
            cons_temp = 0
            cons_max = 0
            for j in range(len(filter_cloud)):
                if filter_cloud[j]:
                    cons_temp += 1
                else:
                    cons_temp = 0
                if cons_temp > cons_max:
                    cons_max = cons_temp

            if cons_max >= self.n_consecutive:
                id_gap = np.append(id_gap, id_values)
                days_gap = np.append(days_gap, days_values)
                long_gap = np.append(long_gap, long_values)
                lati_gap = np.append(lati_gap, lati_values)
                vi_gap = np.append(vi_gap, vi_values)
                cloud_gap = np.append(cloud_gap, cloud_values)

        # STEP 2. Creating target interval
        count = 1
        unique_int = np.empty(0, dtype=int)
        id_int = np.empty(0, dtype=object)
        days_int = np.empty(0, dtype=int)
        long_int = np.empty(0, dtype=float)
        lati_int = np.empty(0, dtype=float)
        vi_int = np.empty(0, dtype=float)
        cloud_int = np.empty(0, dtype=float)

        for i in tqdm(np.unique(id_gap), desc="STEP 2. Creating target interval"):
            mask = (id_gap == i)
            id_values = id_gap[mask]
            days_values = days_gap[mask]
            long_values = long_gap[mask]
            lati_values = lati_gap[mask]
            vi_values = vi_gap[mask]
            cloud_values = cloud_gap[mask]

            filter_cloud = (cloud_values <= self.threshold_cloudy)
            cons_temp = 0
            cons_max = 0

            for j in range(len(filter_cloud)):
                if (filter_cloud[j]) and (j < len(filter_cloud) - 1):
                    cons_temp += 1
                elif filter_cloud[j] and (j == len(filter_cloud) - 1):
                    cons_max = cons_temp + 1
                    cons_temp = 0

                    if cons_max >= self.n_consecutive:
                        if j - cons_max - self.n_tail < 0:
                            id_temp = id_values[0:j + self.n_tail + 1]
                            days_temp = days_values[0:j + self.n_tail + 1]
                            long_temp = long_values[0:j + self.n_tail + 1]
                            lati_temp = lati_values[0:j + self.n_tail + 1]
                            vi_temp = vi_values[0:j + self.n_tail + 1]
                            cloud_temp = cloud_values[0:j + self.n_tail + 1]
                        elif j + self.n_tail >= len(filter_cloud):
                            id_temp = id_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            days_temp = days_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                        else:
                            id_temp = id_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            days_temp = days_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]

                        unique_temp = np.empty(len(id_temp), dtype=int)
                        unique_temp.fill(count)
                        count += 1

                        id_int = np.append(id_int, id_temp)
                        unique_int = np.append(unique_int, unique_temp)
                        days_int = np.append(days_int, days_temp)
                        long_int = np.append(long_int, long_temp)
                        lati_int = np.append(lati_int, lati_temp)
                        vi_int = np.append(vi_int, vi_temp)
                        cloud_int = np.append(cloud_int, cloud_temp)
                        cons_max = 0

                else:
                    cons_max = cons_temp
                    cons_temp = 0

                    if cons_max >= self.n_consecutive:
                        if j - cons_max - self.n_tail < 0:
                            id_temp = id_values[0:j + self.n_tail + 1]
                            days_temp = days_values[0:j + self.n_tail + 1]
                            long_temp = long_values[0:j + self.n_tail + 1]
                            lati_temp = lati_values[0:j + self.n_tail + 1]
                            vi_temp = vi_values[0:j + self.n_tail + 1]
                            cloud_temp = cloud_values[0:j + self.n_tail + 1]
                        elif j + self.n_tail >= len(filter_cloud):
                            id_temp = id_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            days_temp = days_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:len(filter_cloud)]
                        else:
                            id_temp = id_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            days_temp = days_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            long_temp = long_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            lati_temp = lati_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            vi_temp = vi_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]
                            cloud_temp = cloud_values[j - cons_max - self.n_tail + 1:j + self.n_tail + 1]

                        unique_temp = np.empty(len(id_temp), dtype=int)
                        unique_temp.fill(count)
                        count += 1

                        id_int = np.append(id_int, id_temp)
                        unique_int = np.append(unique_int, unique_temp)
                        days_int = np.append(days_int, days_temp)
                        long_int = np.append(long_int, long_temp)
                        lati_int = np.append(lati_int, lati_temp)
                        vi_int = np.append(vi_int, vi_temp)
                        cloud_int = np.append(cloud_int, cloud_temp)
                        cons_max = 0

        # STEP 3. Spatiotemporal filling
        for i in tqdm(np.unique(unique_int), desc="STEP 3. Spatiotemporal filling"):
            mask_int = (unique_int == i)
            id_target = id_int[mask_int]
            days_target = days_int[mask_int]
            long_target = long_int[mask_int]
            lati_target = lati_int[mask_int]
            vi_target = vi_int[mask_int]
            cloud_target = cloud_int[mask_int]

            vi_cand_pred_all = np.empty(0, dtype=float)
            weight_cand_pred_all = np.empty(0, dtype=float)
            distance_cand_all = np.empty(0, dtype=float)
            corr_cand_all = np.empty(0, dtype=float)
            vi_cand_all = np.empty(0, dtype=float)
            filter_cand_all = np.empty(0, dtype=int)
            filterpred_cand_all = np.empty(0, dtype=int)

            # distance to all ids
            distance_values = np.empty(len(idsamp_unique), dtype=float)
            distance_values.fill(0.0000001)

            loc_target = np.array((lati_target[0], long_target[0]))
            filter_target_lo = (cloud_target <= self.threshold_cloudy)
            filter_target_hi = (cloud_target > self.threshold_cloudy)

            for idx, sid in enumerate(idsamp_unique):
                loc_cand = np.array((lat_by_id[sid], lon_by_id[sid]))
                distance_temp = math.dist(loc_target, loc_cand)
                if distance_temp != 0:
                    distance_values[idx] = distance_temp

            distance_invers = np.reciprocal(distance_values)
            distance_norm = np.abs(
                (distance_invers - np.min(distance_invers)) /
                (np.max(distance_invers) - np.min(distance_invers))
            )
            distance_norm[distance_norm <= 0] = 0.0000001

            index_distance = np.argsort(distance_norm)
            base_ids = idsamp_unique[index_distance]

            # remove the target id itself from candidate list
            base_ids = base_ids[base_ids != id_target[0]]

            # Build candidate order based on sampling strategy and id_nested
            if candidate_sampling == "random":
                if nested_by_id is not None:
                    # group by nested id in distance order, then shuffle within each group
                    groups = {}
                    seg_order = []
                    for sid in base_ids:
                        seg = nested_by_id[sid]
                        if seg not in groups:
                            groups[seg] = []
                            seg_order.append(seg)
                        groups[seg].append(sid)
                    cand_list = []
                    for seg in seg_order:
                        arr = np.array(groups[seg])
                        arr = np.random.permutation(arr)
                        cand_list.append(arr)
                    if len(cand_list) > 0:
                        cand_ids = np.concatenate(cand_list)
                    else:
                        cand_ids = np.array([], dtype=base_ids.dtype)
                else:
                    cand_ids = np.random.permutation(base_ids)
            else:
                # "distance" or unknown -> distance-based order
                cand_ids = base_ids

            # Limit candidate pool size if requested
            if (max_candidate_pool is not None) and (len(cand_ids) > max_candidate_pool):
                cand_ids = cand_ids[:max_candidate_pool]

            # Track how many candidates used per nested group (if any)
            nested_counts = {}  # seg -> count of accepted candidates
            # Loop over candidate ids
            for k in cand_ids:
                if (nested_by_id is not None) and (self.n_candidate_nested is not None):
                    seg_k = nested_by_id[k]
                    if nested_counts.get(seg_k, 0) >= self.n_candidate_nested:
                        continue

                mask_cand = (id_sample == k)
                vi_cand = vi_raw[mask_cand]
                cloud_cand = cloud_data[mask_cand]

                distance_cand = distance_norm[idsamp_unique == k]
                last_row = len(vi_cand)
                filter_cand_hi = (cloud_cand > self.threshold_cloudy)
                first_row = 0
                end_row = len(vi_target)
                corr_temp = 0.0
                corr_value = 0.0  # for adaptive step

                # sliding-window search: pick the window with max correlation
                while end_row <= last_row:
                    filter_hi = filter_target_hi * filter_cand_hi[first_row:end_row]
                    filter_pred = filter_target_lo * filter_cand_hi[first_row:end_row]

                    if np.sum(filter_hi) >= np.sum(filter_target_hi) / 2 and np.sum(filter_pred) > 0:
                        vi_cand_cut = vi_cand[first_row:end_row]
                        corr_value = np.corrcoef(vi_target[filter_hi], vi_cand_cut[filter_hi])[0, 1]
                        if corr_value > corr_temp:
                            corr_temp = corr_value
                            vi_cand_temp = vi_cand_cut
                            filter_temp = filter_hi
                            filter_pred_temp = filter_pred

                    corr_use = max(corr_temp, corr_value)
                    corr_use = max(0.0, min(1.0, corr_use))  # clamp to [0,1]

                    step = int(round(self.step_max - corr_use * (self.step_max - self.step_min)))
                    step = max(self.step_min, min(step, self.step_max))

                    first_row += step
                    end_row += step

                if corr_temp >= self.threshold_corr:
                    distance_cand_all = np.append(distance_cand_all, distance_cand)
                    corr_cand_all = np.append(corr_cand_all, corr_temp)
                    vi_cand_all = np.append(vi_cand_all, vi_cand_temp)
                    filter_cand_all = np.append(filter_cand_all, filter_temp)
                    filterpred_cand_all = np.append(filterpred_cand_all, filter_pred_temp)
                    
                if (self.n_candidate is not None) and (len(corr_cand_all) == self.n_candidate):
                    break

            # Combine candidates
            for l in range(len(corr_cand_all)):
                distance_cand_temp = distance_cand_all[l]
                corr_cand_temp = corr_cand_all[l]
                vi_cand_temp = vi_cand_all[l * len(vi_target):(l + 1) * len(vi_target)]
                filter_cand_temp = filter_cand_all[l * len(vi_target):(l + 1) * len(vi_target)]
                filterpred_cand_temp = filterpred_cand_all[l * len(vi_target):(l + 1) * len(vi_target)]

                model_coef = np.polyfit(
                    x=vi_cand_temp[filter_cand_temp == 1],
                    y=vi_target[filter_cand_temp == 1],
                    deg=3,
                )
                model_pred = np.poly1d(model_coef)
                vi_cand_pred = model_pred(vi_cand_temp)

                if self.vi_max is not None:
                    vi_cand_pred[vi_cand_pred > self.vi_max] = self.vi_max
                if self.vi_min is not None:
                    vi_cand_pred[vi_cand_pred < self.vi_min] = self.vi_min

                vi_cand_pred[filterpred_cand_temp == 0] = 0

                vi_cand_pred_fin = vi_cand_pred * corr_cand_temp * distance_cand_temp
                weight_cand_pred = filterpred_cand_temp * corr_cand_temp * distance_cand_temp
                weight_cand_pred[filterpred_cand_temp == 0] = 0

                vi_cand_pred_all = np.append(vi_cand_pred_all, vi_cand_pred_fin)
                weight_cand_pred_all = np.append(weight_cand_pred_all, weight_cand_pred)

            if len(vi_cand_pred_all) > 0:
                vi_pred_sum = np.nansum(
                    np.split(vi_cand_pred_all, int(len(vi_cand_pred_all) / len(vi_target))),
                    axis=0,
                )
                filterpred_sum = np.nansum(
                    np.split(filterpred_cand_all, int(len(filterpred_cand_all) / len(vi_target))),
                    axis=0,
                )
                weight_pred_sum = np.nansum(
                    np.split(weight_cand_pred_all, int(len(weight_cand_pred_all) / len(vi_target))),
                    axis=0,
                )

                vi_pred_fin = np.divide(
                    vi_pred_sum,
                    weight_pred_sum,
                    out=vi_target.copy(),          
                    where=weight_pred_sum > 0,
                )

                if (self.vi_max is not None) or (self.vi_min is not None):
                    bad_mask = np.zeros_like(vi_pred_fin, dtype=bool)
                    if self.vi_max is not None:
                        bad_mask |= vi_pred_fin > self.vi_max
                    if self.vi_min is not None:
                        bad_mask |= vi_pred_fin < self.vi_min
                    vi_pred_fin[bad_mask] = vi_target[bad_mask]

                for m in days_target[filterpred_sum > 0]:
                    vi_data[
                        np.where(
                            (id_sample == id_target[0]) & (days_data == m)
                        )[0]
                    ] = vi_pred_fin[days_target == m]

        return vi_data

    # ------------------------------------------------------------------
    # multistep_smoothing
    # ------------------------------------------------------------------
    def multistep_smoothing(self, id_sample, days_data, vi_data, cloud_data):
        """
        Applies iterative GAM-based smoothing on each time series using progressively stricter
        cloud confidence weights.
        """
        idsamp_unique = np.unique(id_sample)
        cloud_init = cloud_data.copy()
        for i in tqdm(idsamp_unique, desc="Multistep Smoothing"):
            days_values = days_data[id_sample == i]
            vi_values = vi_data[id_sample == i]
            cloud_init_sample = cloud_init[id_sample == i]
            cloud_values = cloud_data[id_sample == i]
            for j in np.arange(
                self.smoothing_min,
                self.smoothing_max + self.smoothing_increment,
                self.smoothing_increment,
            ):
                gam = LinearGAM(s(0), n_splines=self.n_spline).fit(
                    days_values, vi_values, weights=cloud_values
                )
                gam.gridsearch(
                    days_values,
                    vi_values,
                    weights=cloud_values,
                    lam=self.lamdas,
                    objective="GCV",
                    progress=False,
                )
                vi_smooth = gam.predict(days_values)
                vi_diff = vi_smooth - vi_values

                if self.vi_max is not None:
                    vi_smooth[vi_smooth > self.vi_max] = self.vi_max
                if self.vi_min is not None:
                    vi_smooth[vi_smooth < self.vi_min] = self.vi_min

                cloud_new = np.abs(
                    (vi_diff - np.min(vi_diff))
                    / (np.max(vi_diff) - np.min(vi_diff))
                )
                cloud_new[cloud_new == 0] = 0.0000001
                cloud_new[
                    (cloud_init_sample > j) & (cloud_new <= cloud_init_sample)
                ] = cloud_init_sample[
                    (cloud_init_sample > j) & (cloud_new <= cloud_init_sample)
                ]
                cloud_new[
                    (cloud_init_sample <= j) & (cloud_new > cloud_init_sample)
                ] = cloud_init_sample[
                    (cloud_init_sample <= j) & (cloud_new > cloud_init_sample)
                ]
                vi_values[cloud_init_sample <= j] = vi_smooth[cloud_init_sample <= j]
                cloud_values = cloud_new
            vi_data[id_sample == i] = vi_values
        return vi_data
