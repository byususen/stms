# STMS: Spatiotemporal and Multistep Smoothing for Sentinel-2 Data Reconstruction

STMS is a Python package designed to reconstruct and smooth time-series vegetation index (VI) data, particularly useful in handling cloudy observations in satellite imagery like Sentinel-2.

It performs two main steps:
1. **Spatiotemporal Filling** — Uses spatial neighbors and correlation to fill in cloudy or missing data.
2. **Multistep Smoothing** — Applies Generalized Additive Models (GAMs) for smoothing over time.

---

## 📦 Installation

```bash
pip install stms
```

---

## 🔬 Features

- Handles **consecutive cloudy observations**
- Incorporates **spatial proximity** and **temporal correlation**
- Multi-round **GAM-based smoothing**
- Easy-to-use API

---

## 🧪 Example: Simulated Sentinel-2 Time Series

```python
import numpy as np
from stms import stms

# Simulate vegetation index (VI) using sine function
def sine_func(x, A, B, C, D):
    return A * np.sin(2 * (np.pi / B) * (x - C)) + D

# Time series parameters
A, B, C, D = 0.3, 100, 90, 0.5
x = np.arange(5, 400, 5)
vi = sine_func(x, A, B, C, D) + np.random.uniform(-0.05, 0.05, len(x))
cloud = np.ones_like(vi)

# Add thick cloud contamination
vi[50:60] = np.random.uniform(0.1, 0.2, 10)
cloud[50:60] = 0.01

# Format for STMS
id_sample = np.array(["sample_0"] * len(x))
days_data = x
vi_data = vi.copy()
long_data = np.array([101.5] * len(x))
lati_data = np.array([-2.0] * len(x))
cloud_data = cloud

# Apply STMS
model = stms()
vi_filled = model.spatiotemporal_filling(id_sample, days_data, vi_data, long_data, lati_data, cloud_data)
vi_smoothed = model.multistep_smoothing(id_sample, days_data, vi_filled, cloud_data)
```

---

## 📈 Visual Output (1 Sample)

### Original (Cloudy)
![Original](examples/simulated_original.png)

### After STMS Filling
![Filled](examples/simulated_filled.png)

### Final Smoothed Result
![Final](examples/simulated_final_result.png)

---

## 📄 License

MIT License © Bayu Suseno

---

## 🤝 Contributing

Feel free to open issues or pull requests! Contributions are welcome.
