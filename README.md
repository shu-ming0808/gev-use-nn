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

### 6. Spatial Simulation Validation

To check whether the NN + kriging framework can recover a known spatial GEV structure, this project also includes a simulation study using the same 25 station coordinates as the real Taiwan data.

The simulation workflow is:

```text
station coordinates
→ known spatial GEV parameter fields
→ simulated 45-year annual maxima and 45 × 12 monthly maxima
→ NN station-level parameter estimation for both sample sizes
→ RBF and Matérn Gaussian-process kriging
→ comparison with known truth
```

The known parameter fields are generated as smooth spatial functions:

$$
\mu(s) = \beta_0 + \beta_1 lon(s) + \beta_2 lat(s) + W_\mu(s)
$$

$$
\log\sigma(s) = \gamma_0 + \gamma_1 lon(s) + \gamma_2 lat(s) + W_\sigma(s)
$$

$$
\xi(s) = \alpha_0 + \alpha_1 lat(s) + W_\xi(s)
$$

where each $W(s)$ is generated from a Gaussian process with an RBF covariance function.

For the kriging step, both RBF and Matérn kernels are generated. The RBF kernel assumes a very smooth spatial field:

$$
k_{RBF}(s,s') = \sigma_f^2
\exp\left(-\frac{\|s-s'\|^2}{2\ell^2}\right)
$$

The Matérn kernel is also included because it is commonly used in spatial statistics and allows less smooth spatial variation. In this project, the Matérn kernel uses $\nu = 1.5$:

$$
k_{Matérn}(s,s') =
\sigma_f^2
\left(1+\frac{\sqrt{3}d}{\ell}\right)
\exp\left(-\frac{\sqrt{3}d}{\ell}\right),
\quad d=\|s-s'\|
$$

The length scale is initialized at 1 after coordinate standardization and is optimized by Gaussian-process marginal likelihood within the specified bounds.

Run:

```bash
python src/simulate_spatial_gev.py
```

Outputs are saved under:

```text
data/simulated/spatial_gev/
```

Key outputs:

- `spatial_station_true_params.csv`: known true station-level GEV parameters
- `spatial_annual_max_25stations.csv`: simulated 45-year annual maxima
- `spatial_monthly_max_25stations.csv`: simulated 45-year monthly maxima, about 540 samples per station
- `spatial_station_nn_estimates.csv`: NN-estimated station-level parameters
- `spatial_station_true_vs_nn.csv`: station-level comparison
- `spatial_grid_true_params.csv`: true spatial fields on a grid
- `spatial_grid_nn_rbf_kriging_params.csv`: NN + RBF kriging estimated spatial fields
- `spatial_grid_nn_matern_kriging_params.csv`: NN + Matérn kriging estimated spatial fields
- `spatial_grid_nn_kriging_params.csv`: backward-compatible RBF kriging output
- `spatial_station_error_summary.csv`: RMSE, MAE, and correlation
- `spatial_grid_error_summary.csv`: grid-level RMSE, MAE, and correlation for RBF vs Matérn kriging
- `spatial_annual_*`: annual maxima analysis outputs
- `spatial_monthly_*`: monthly maxima sensitivity analysis outputs
- `spatial_annual_monthly_grid_error_summary.csv`: annual vs monthly grid-level comparison

The final cells in `notebooks/main.ipynb` visualize the true spatial fields, the NN + RBF kriging estimates, and the NN + Matérn kriging estimates as clipped Taiwan maps. Each map is arranged as a 1 × 3 panel for $\mu$, $\sigma$, and $\xi$, followed by a table comparing RBF and Matérn by grid-level error.

### 7. Annual vs Monthly Sensitivity Analysis

The annual analysis uses one block maximum per year:

```text
annual: 45 samples per station
```

The monthly sensitivity analysis uses one block maximum per month:

```text
monthly: about 540 samples per station
```

The purpose is not to replace the annual extreme-value analysis, but to test whether a larger number of block maxima makes the NN-estimated station parameters more stable before kriging. This is especially useful for $\sigma$ and $\xi$, because these parameters are usually harder to estimate from only 45 annual maxima.

In `notebooks/main.ipynb`, the sensitivity section is split into:

- data preparation: generate and load annual/monthly simulated maxima
- calculation and visualization: plot true fields, annual NN + RBF kriging, and monthly NN + RBF kriging using the same 1 × 3 Taiwan map layout

The comparison table reports RMSE, MAE, and correlation for annual and monthly results under both RBF and Matérn kriging.

---

## Project Structure

```text
fast_parameter_using_NN/
│
├── data/
│   ├── original_data/
│   ├── processed/
│   └── simulated/
│
├── src/
│   ├── prepare_annual_max.py
│   ├── estimate_real_params.py
│   ├── merge_station_data.py
│   ├── kriging_params.py
│   ├── plot_gev_maps.py
│   ├── compute_return_level.py
│   ├── simulate_data.py
│   ├── simulate_spatial_gev.py
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

For the spatial simulation validation:

```bash
python src/simulate_spatial_gev.py
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
