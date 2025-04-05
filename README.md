# STMS-Filler

Spatiotemporal Filling and Multistep Smoothing (STMS) is a method to reconstruct satellite time series affected by cloud contamination. This tool helps recover missing values based on spatial similarity and temporal smoothing.

## Installation

```bash
pip install git+https://github.com/byususen/stms.git
```

## Usage

```python
from stms_filler import stms

filler = stms()
vi_filled = filler.spatiotemporal_filling(...)
vi_smoothed = filler.multistep_smoothing(...)
```
