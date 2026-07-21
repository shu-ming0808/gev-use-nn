# Six directional RBF–Matérn kernel tests

## Outcomes

- Original stations: annual and monthly parameter recovery are tested
  separately, with Matérn as the pre-specified directional alternative.
- Gridded data: annual and monthly parameter recovery are tested separately,
  with RBF as the pre-specified directional alternative.
- Gridded monthly data: annual 50-year and 100-year return-level recovery are
  tested separately, with RBF as the pre-specified directional alternative.

## Test design

- These are six pre-specified, one-sided exact paired sign-flip tests.
- The unit of inference is one of ten geographic K-means blocks.
- Parameter-recovery loss jointly averages standardized squared errors for
  $\mu$, $\sigma$, and $\xi$ without pooling annual and monthly scenarios.
- Return-level loss is squared prediction error in the temperature unit.
- Individual adjacent grid cells are not treated as independent replicates.
- With ten blocks, the minimum attainable one-sided p-value is
  $1/2^{10}=0.0009765625$.
- RBF and Matérn $\nu=0.5$ are fixed before validation. GP hyperparameters
  are estimated by marginal likelihood from training responses only.
- Gridded predictions are spatial out-of-fold: each block is predicted from
  the other nine blocks.
- Holm adjustment controls family-wise error across all six hypotheses.

## Results

| Outcome | Loss scale | RBF RMSE | Matérn RMSE | Directional $H_1$ | Raw $p$ | Holm $p$ | Decision |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| original_station_annual | standardized parameters | 1.758348 | 1.752143 | Matern | 0.397461 | 1.000000 | do not reject H0 |
| original_station_monthly | standardized parameters | 0.793053 | 0.585121 | Matern | 0.000977 | 0.005859 | reject H0 |
| gridded_annual | standardized parameters | 1.696226 | 1.665217 | RBF | 0.839844 | 1.000000 | do not reject H0 |
| gridded_monthly | standardized parameters | 0.988339 | 0.912764 | RBF | 0.990234 | 1.000000 | do not reject H0 |
| gridded_monthly_RL50 | temperature | 2.188238 | 1.953513 | RBF | 0.935547 | 1.000000 | do not reject H0 |
| gridded_monthly_RL100 | temperature | 2.590963 | 2.261655 | RBF | 0.960938 | 1.000000 | do not reject H0 |
