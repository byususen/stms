import numpy as np
import matplotlib.pyplot as plt
from stms_filler import stms

# Generate synthetic time series data
def sine_func(x, A, B, C, D):
    return A * np.sin(2 * (np.pi / B) * (x - C)) + D

A, B = 0.3, 100
n_sample, n_days = 50, 400
x = np.arange(5, n_days, 5)

vi_data, long_data, lati_data, cloud_data = np.array([]), np.array([]), np.array([]), np.array([])
days_data, id_sample = np.array([], dtype=int), np.array([], dtype=object)

for i in range(n_sample):
    long = np.random.uniform(100, 102)
    lati = np.random.uniform(-3, -1)
    idsamp = f"sample_{i}"
    C = 100 + np.random.uniform(-50, 50)
    D = 0.5 + np.random.uniform(-0.05, 0.05)
    for j in x:
        y = sine_func(j, A, B, C, D) + np.random.uniform(-0.1, -0.01)
        cloud = 0.9 + np.random.uniform(-0.1, -0.01)
        vi_data = np.append(vi_data, y)
        cloud_data = np.append(cloud_data, cloud)
        long_data = np.append(long_data, long)
        lati_data = np.append(lati_data, lati)
        days_data = np.append(days_data, int(j))
        id_sample = np.append(id_sample, idsamp)

# Add random cloudy days
for i in np.unique(id_sample):
    loc = np.random.choice(x, 5, replace=False)
    for j in loc:
        idx = np.where((id_sample == i) & (days_data == j))[0]
        vi_data[idx] = np.random.uniform(0.1, 0.2)
        cloud_data[idx] = np.random.uniform(0.0, 0.1)

# Add consecutive cloudy segment
for i in np.unique(id_sample):
    length_cons = np.random.randint(5, 10)
    loc_first = np.random.choice(np.arange(0, len(id_sample[id_sample == i]) - length_cons), 1)[0]
    loc_days = x[int(loc_first):int(loc_first + length_cons)]
    for j in loc_days:
        idx = np.where((id_sample == i) & (days_data == j))[0]
        vi_data[idx] = np.random.uniform(0.1, 0.2)
        cloud_data[idx] = np.random.uniform(0.0, 0.1)

# Apply STMS
stms_method = stms()
vi_spa = stms_method.spatiotemporal_filling(id_sample, days_data, vi_data, long_data, lati_data, cloud_data)
vi_stms = stms_method.multistep_smoothing(id_sample, days_data, vi_spa, cloud_data)
