# Fast Parameter Estimation of GEV using Neural Networks

## Project Overview

This project implements a fast estimation framework for the Generalized Extreme Value (GEV) distribution using neural networks, and extends it to spatial modeling using kriging.

The main objectives are:

* Estimate GEV parameters (μ, σ, ξ) efficiently
* Apply the model to real-world Taiwan climate data (TCCIP)
* Model spatial dependence using Gaussian processes
* Generate spatial maps and return level surfaces

Based on:
"Fast parameter estimation of generalized extreme value distribution using neural networks"

---

## Methodology

### 1. Neural Network Estimation

Instead of traditional MLE, we learn:

Quantiles → (μ, σ, ξ)

Neural network is trained on simulated GEV data and predicts parameters directly.

---

### 2. Real Data

* Data source: TCCIP Taiwan climate data
* 25 weather stations
* 45 years of annual maxima

Shape:
(45 × 25)

---

### 3. Spatial Extension (Core Idea)

Original paper only estimates parameters at each station independently.

We extend it to spatial modeling:

μ(s) = x(s)^T β + W(s)

Where:

* x(s): spatial features (lon, lat, etc.)
* W(s): Gaussian process

Interpretation:
parameter = trend + spatial dependence

---

### 4. Two-Stage Framework

Step 1:
NN estimates station-level parameters

θ̂(s_i) = (μ̂, σ̂, ξ̂)

Step 2:
Kriging interpolates spatial field

NN → stations → kriging → full map

---

### 5. Final Output

We obtain:

* Spatial maps:

  * μ(s)
  * σ(s)
  * ξ(s)

* Return level surface:

z_T(s) = μ(s) + σ(s)/ξ(s) * [(-log(1 - 1/T))^{-ξ(s)} - 1]

Meaning:
Extreme value expected once every T years

---

## Project Structure

fast_parameter_using_NN/
│
├── data/
│   ├── original_data/
│   └── processed/
│
├── src/
│   ├── prepare_annual_max.py
│   ├── estimate_real_params.py
│   ├── merge_station_data.py
│   ├── kriging_params.py
│   ├── plot_gev_maps.py
│   ├── compute_return_level.py
│
├── notebooks/
│   └── main.ipynb
│
├── results/
│   └── figures/
│
├── models/
└── README.md

---

## Pipeline

Run in order:

python src/prepare_annual_max.py
python src/estimate_real_params.py
python src/merge_station_data.py
python src/kriging_params.py
python src/plot_gev_maps.py
python src/compute_return_level.py

Or run everything in:

notebooks/main.ipynb

---

## Results

Outputs include:

* Spatial distribution of μ, σ, ξ
* Taiwan clipped maps
* Return level maps (e.g., 100-year extreme value)

---

## Key Contributions

* Fast GEV estimation using neural networks
* Bootstrap-based uncertainty estimation
* Extension to spatial modeling (NN + Kriging)
* Real-world application (Taiwan climate data)

---

## Limitations

* NN outputs may violate constraints (σ > 0)
* Spatial model assumes Gaussian process
* Limited number of stations (25)

---

## Future Work

* Spatio-temporal GEV modeling
* CNN-based spatial learning
* Direct return level prediction
* Larger spatial datasets

---

## Author

Shu-Ming Chang
National Central University
