# EAS paired simulation summary

Source: four completed NN and four completed MLE runs under
`~/Desktop/模擬資料/{nn,mle}/replicate_001..004/attempt_001`.
Each replicate has matching NN/MLE sample IDs; the generating truth is the
same 7,822-grid five-coefficient surface for all four replicates.

- The replicate-level RMSE is calculated over the 7,822 grids. Mean, sample
  SD, sample variance and two-sided 95% Student-t CI (df = 3) are then
  calculated over the four replicates. A CI here is conditional on the fixed
  simulated truth, EAS grid and CV design; it is not a per-grid uncertainty
  interval or a claim about all possible climates.
- `direct` means NN/MLE coefficient estimates. `gp_oof` means predictions
  on outer spatial-CV held-out grids.
- Standardized RMSE divides each coefficient's RMSE by the spatial SD of
  that coefficient in the fixed generating truth. It does not rescale
  predictions or alter the model.
- The old single-run comparison under Desktop `simulations` uses the same
  generating truth and numerically the same first sequence (maximum absolute
  difference about 7e-15 °C) as paired replicate 001. It has a different
  protocol ID and must not be counted as a fifth independent replicate.
- The five coefficient maps repeat across years 1, 30 and 50 because the
  coefficients themselves do not change with year. The supplementary
  `mu_t` and `sigma_t` maps show the quantities that do change. Map rows
  are generating truth, four-replicate mean NN + GP OOF and four-replicate
  mean MLE + GP OOF; all nine panels within a figure use one color scale.

Regenerate from the repository root:

```powershell
python scripts/analyze_eas_paired_simulations.py
```

Upload the desired PNGs alongside `eas_simulation_tables.tex` and
`eas_simulation_maps.tex` to Overleaf; the deck needs `booktabs` and
`graphicx`.
