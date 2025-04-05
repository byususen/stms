"""
Simulated Data Example for STMS Package

This script:
1. Generates synthetic Sentinel-2-like vegetation index time series
2. Adds cloud contamination
3. Applies STMS reconstruction and smoothing
4. Saves before/after/final result plots
"""

import numpy as np
import matplotlib.pyplot as plt
from stms import stms

# === Generate Simulated Data === #
def sine_func(x, A, B, C, D):
    return A * np.sin(2 * (np.pi / B) * (x - C)) + D

A, B, C, D = 0.3, 100, 90, 0.5
x = np.arange(5, 400, 5)
vi_clean = sine_func(x, A, B, C, D)
vi = vi_clean + np.random.uniform(-0.05, 0.05, len(x))
cloud = np.ones_like(vi)

# Inject cloud contamination
vi[50:60] = np.random.uniform(0.1, 0.2, 10)
cloud[50:60] = 0.01

# Create STMS-compatible input arrays
n = len(x)
id_sample = np.array(["sample_0"] * n)
days_data = x
vi_data = vi.copy()
long_data = np.array([101.5] * n)
lati_data = np.array([-2.0] * n)
cloud_data = cloud

# === Apply STMS === #
model = stms()
vi_filled = model.spatiotemporal_filling(id_sample, days_data, vi_data, long_data, lati_data, cloud_data)
vi_smoothed = model.multistep_smoothing(id_sample, days_data, vi_filled, cloud_data)

# === Save Visual Outputs === #
# Original
plt.figure(figsize=(10, 4))
plt.plot(x, vi, 'o-', alpha=0.4, label="Original (Cloudy)")
plt.title("Original Time Series with Cloud Contamination")
plt.xlabel("Days")
plt.ylabel("Vegetation Index")
plt.grid(True)
plt.tight_layout()
plt.savefig("examples/simulated_original.png")
plt.close()

# After STMS Fill
plt.figure(figsize=(10, 4))
plt.plot(x, vi, 'o-', alpha=0.4, label="Cloudy")
plt.plot(x, vi_filled, 's-', alpha=0.7, label="STMS Filled")
plt.title("After STMS Gap Filling")
plt.xlabel("Days")
plt.ylabel("Vegetation Index")
plt.grid(True)
plt.tight_layout()
plt.savefig("examples/simulated_filled.png")
plt.close()

# Final STMS Result
plt.figure(figsize=(10, 4))
plt.plot(x, vi, 'o-', alpha=0.3, label="Cloudy")
plt.plot(x, vi_filled, 's-', alpha=0.6, label="Filled")
plt.plot(x, vi_smoothed, 'd-', alpha=0.9, label="Smoothed")
plt.title("Final STMS Result")
plt.xlabel("Days")
plt.ylabel("Vegetation Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("examples/simulated_final_result.png")
plt.close()
