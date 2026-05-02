# GEV Neural Estimation and Spatial Extension

This project reproduces a neural network estimator for GEV parameters
and extends it to spatial modeling using kriging / Gaussian processes.

---

## Project Structure

- `src/` : all scripts (training, simulation, bootstrap, MLE)
- `data/` :
  - raw : original data
  - interim : simulated samples
  - processed : results (NN, MLE, bootstrap)
- `models/` : trained NN models
- `notebooks/` : experiments and real data analysis
- `results/` : plots and outputs
- `reports/` : slides and papers

---

## Workflow

1. Generate simulated GEV data
2. Train NN model (baseline / weighted)
3. Compare with MLE
4. Apply NN to real station data
5. Extend to spatial parameter process (kriging)

---

## Key Idea

Instead of estimating GEV parameters independently,
we treat μ(s), σ(s), ξ(s) as spatial random processes.

NN → estimate parameters at stations  
Kriging → interpolate across space