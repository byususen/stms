# 🌿 STMS: Spatiotemporal and Multistep Smoothing for Time-Series Reconstruction

**STMS** is a Python package for reconstructing and smoothing cloud-affected or noisy time-series data, particularly from satellite-derived vegetation indices (VI) such as NDVI, EVI, MSAVI, NIRv, and other time-series environmental indicators

Although originally designed for Sentinel-2 data, STMS is data-agnostic and works with any temporal signal measured at geolocated sampling units.

---

## ✨ What STMS Does

STMS performs two complementary steps:

1️⃣ Spatiotemporal Filling

Reconstructs missing or low-quality values by:

Detecting consecutive unreliable observations

Searching for spatially nearby or structurally similar time series

Selecting candidates based on correlation, distance, and/or grouping (nested IDs)

Predicting values through polynomial regression and weighted aggregation

2️⃣ Multistep Smoothing

Applies iterative Generalized Additive Models (GAMs) with increasing quality thresholds to:

Smooth noisy observations

Preserve phenological or seasonal shapes

Adaptively reweight low-confidence points

Produce a final clean, continuous time-series signal

STMS is suitable for:

Cloudy or incomplete satellite vegetation indices

Remote sensing environmental monitoring

Agricultural time-series

Ecological and climate data

Any geotemporal dataset with missing or noisy values

---

## 📦 Installation

From PyPI:

```bash
pip install stms
```

---

## ⚙️ Basic Usage
```python
from stms import stms

model = stms()

vi_filled = model.spatiotemporal_filling(
    id_sample=id_array,
    days_data=day_of_year,
    vi_data=vi_raw,
    long_data=longitude,
    lati_data=latitude,
    cloud_data=cloudscore
)

vi_smoothed = model.multistep_smoothing(
    id_sample=id_array,
    days_data=day_of_year,
    vi_data=vi_filled,
    cloud_data=cloudscore
)
```

### Required Inputs
| Name                     | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| `id_sample`              | ID for each time-series (one ID per pixel/plot/station) |
| `days_data`              | Time axis as day-of-year or numeric timestamp           |
| `vi_data`                | Raw VI values (cloudy allowed)                          |
| `long_data`, `lati_data` | Spatial coordinates (used for candidate search)         |
| `cloud_data`             | CloudScore+ or quality weights                          |


---

### 🔧 Parameter Overview

You can customize the STMS behavior when constructing the model:
```python
model = stms(
    n_consecutive=5,
    threshold_cloudy=0.1,
    threshold_corr=0.9,
    n_candidate=10,
    n_tail=24,
    id_nested=None,
    n_candidate_nested=None,
    max_candidate_pool=None,
    candidate_sampling="distance"
)
```

| Parameter          | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `n_consecutive`    | Minimum consecutive low-quality points considered a gap  |
| `n_tail`           | Padding before/after a gap                               |
| `threshold_cloudy` | Quality threshold to classify an observation as “cloudy” |
| `threshold_corr`   | Minimum correlation required to accept a candidate       |
| `n_candidate`      | Maximum global number of candidate series used           |
| `id_nested`          | Optional grouping (e.g., pixel → field, station → region) |
| `n_candidate_nested` | Limit candidates per group                                |
| `max_candidate_pool` | Maximum groups selected                                   |
| `candidate_sampling` | `"distance"` or `"random"` group sampling                 |

---

## 🧪 Example with Simulated Data

```python
import numpy as np
from stms import stms

# simulate VI curve
def sine_curve(t):
    return 0.3*np.sin(2*np.pi/100 * (t - 90)) + 0.5

x = np.arange(0, 365, 5)
vi = sine_curve(x) + np.random.normal(0, 0.02, len(x))
cloud = np.ones_like(vi)

# introduce synthetic cloud contamination
cloud[40:55] = 0.01
vi[40:55] = np.random.uniform(0.05, 0.1, 15)

model = stms()
vi_filled = model.spatiotemporal_filling(
    id_sample=np.array(["A"]*len(x)),
    days_data=x,
    vi_data=vi,
    long_data=np.repeat(110.0, len(x)),
    lati_data=np.repeat(-7.0, len(x)),
    cloud_data=cloud
)

vi_smooth = model.multistep_smoothing(
    id_sample=np.array(["A"]*len(x)),
    days_data=x,
    vi_data=vi_filled,
    cloud_data=cloud
)

```

---
## 📚 Citation

If STMS contributes to your research, please cite:

Suseno, B., Brunel, G., Wijayanto, H., Sadik, K., Afendi, F. M., & Tisseyre, B. (2025). 
Reconstructing satellite temporal series data under cloudy conditions: Application in predicting rice growth phases. 
Smart Agricultural Technology, 12, 101378. https://doi.org/10.1016/j.atech.2025.101378

---
## 📄 License

MIT License © Bayu Suseno

---
## 🤝 Contributing

Pull requests, bug reports, and feature suggestions are welcome!
Please open an issue if you encounter any problem.
