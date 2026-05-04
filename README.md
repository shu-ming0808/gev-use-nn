# Fast Parameter Estimation of GEV using Neural Networks

## Project Overview

This project implements a fast estimation framework for the Generalized Extreme Value (GEV) distribution using neural networks, and extends it to spatial modeling via kriging.

### Objectives


- Estimate GEV parameters $$ (\mu, \sigma, \xi) $$ efficiently
- Apply the model to Taiwan climate data (TCCIP)
- Model spatial dependence using Gaussian processes
- Generate spatial maps and return level surfaces

Based on:

> *Fast parameter estimation of generalized extreme value distribution using neural networks*

---

## Quick Start

bash
git clone https://github.com/shu-ming0808/gev-use-nn.git
cd gev-use-nn

conda env create -f environment.yml
conda activate gev-nn

python src/prepare_annual_max.py

---

## Methodology

### 1. Neural Network Estimation

Instead of traditional MLE, we learn:

```text
Quantiles → (μ, σ, ξ)
```

The neural network is trained on simulated GEV samples and directly predicts parameters.

---

### 2. Real Data

- Data source: TCCIP Taiwan climate data
- 25 weather stations
- 45 years of annual maxima

```text
Shape: (45 × 25)
```

---
### Shapefile Data

Taiwan boundary data should be placed under:

```text
data/shapefile/ne_50m_admin_0_countries/
```
---

### 3. Spatial Extension (Core Idea)

The original paper estimates parameters independently at each station.

We extend it to spatial modeling:


$$
\mu(s) = x(s)^T \beta + W(s)
$$

where:

- $$x(s)$$: spatial features (longitude, latitude, etc.)
- $$W(s)$$: Gaussian process


**Interpretation**

> parameter = trend + spatial dependence

---

### 4. Two-Stage Framework

**Step 1: Neural Network**

$$
\hat{\theta}(s_i) = (\hat{\mu}, \hat{\sigma}, \hat{\xi})
$$

**Step 2: Kriging**

```text
NN → station estimates → kriging → spatial field
```

---

### 5. Return Level

  * $\mu(s)$
  * $\sigma(s)$
  * $\xi(s)$

* Return level surface:

$$
z_T(s) = \mu(s) + \frac{\sigma(s)}{\xi(s)} 
\left[ \left(-\log\left(1 - \frac{1}{T}\right)\right)^{-\xi(s)} - 1 \right]
$$

Meaning:

> Extreme value expected once every $$T$$ years

---

## Project Structure

```text
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
```

---

## Pipeline

Run scripts in order:

```bash
python src/prepare_annual_max.py
python src/estimate_real_params.py
python src/merge_station_data.py
python src/kriging_params.py
python src/plot_gev_maps.py
python src/compute_return_level.py
```

Or run everything in:

```bash
notebooks/main.ipynb
```

---

## Outputs

- Spatial maps of:
  - $$\mu(s)$$
  - $$\sigma(s)$$
  - $$\xi(s)$$
- Taiwan clipped maps
- Return level maps (e.g., 100-year extreme)

---

## Key Contributions

- Fast GEV estimation via neural networks
- Bootstrap-based uncertainty estimation
- Spatial extension (NN + Kriging)
- Real-world application (Taiwan climate data)

---

## Limitations

- Neural network outputs may violate constraints (e.g., $$\sigma > 0$$)
- Gaussian process assumption for spatial modeling
- Limited number of stations (25)

---

## Future Work

- Spatio-temporal GEV modeling
- CNN-based spatial learning
- Direct return level prediction
- Larger spatial datasets

---

## Author

Shu-Ming Chang  
National Central University