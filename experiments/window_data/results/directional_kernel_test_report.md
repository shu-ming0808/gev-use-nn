# Directional RBF–Matérn kernel tests

## Confirmatory hypotheses

### Test 1: original station input

For geographic block $b$, define

$$
C_b=L_{\mathrm{RBF},b}-L_{\mathrm{Mat\acute{e}rn},b}.
$$

$$
H_0:E(C_b)\leq 0,
\qquad
H_1:E(C_b)>0.
$$

The one-sided alternative means that Matérn has smaller joint standardized
recovery loss than RBF.

Result: Matern RMSE = 1.428071; one-sided exact p = 0.179688; do not reject H0: insufficient evidence that Matern is better.

### Test 2: gridded input

For geographic block $b$, define

$$
C_b=L_{\mathrm{Mat\acute{e}rn},b}-L_{\mathrm{RBF},b}.
$$

$$
H_0:E(C_b)\leq 0,
\qquad
H_1:E(C_b)>0.
$$

The one-sided alternative means that RBF has smaller joint standardized
recovery loss than Matérn.

Result: RBF RMSE = 2.052587; one-sided exact p = 0.000977; reject H0: RBF is significantly better.

## Test design

- These are two pre-specified, one-sided exact paired sign-flip tests.
- The unit of inference is one of ten geographic K-means blocks.
- Each block loss jointly averages standardized squared recovery errors for
  $\mu$, $\sigma$, and $\xi$ in both annual-45 and monthly-540 simulations.
- Individual adjacent grid cells are not treated as independent replicates.
- With ten blocks, the minimum attainable one-sided p-value is
  $1/2^{10}=0.0009765625$.
- The original station-input experiment uses the saved exhaustive fixed-kernel
  grid search in `spatial_kernel_gridsearch_rmse.csv`.
- The gridded-input experiment exactly reproduces the notebook table, including
  the original pretrained-network inverse transform and RNG ordering.
- The tests assess the two stated directional claims. They are not two-sided
  generic difference tests, and AIC is not used as a p-value.

## Descriptive RMSE values reproduced from the two tables

| Experiment | Scenario | Parameter | Kernel | RMSE | Metric |
| --- | --- | --- | --- | ---: | --- |
| original_station_input | annual | mu | Matern | 2.078087 | RMSE |
| original_station_input | annual | mu | RBF | 2.471700 | RMSE |
| original_station_input | annual | sigma | Matern | 3.870187 | RMSE |
| original_station_input | annual | sigma | RBF | 3.925145 | RMSE |
| original_station_input | annual | xi | Matern | 0.041339 | RMSE |
| original_station_input | annual | xi | RBF | 0.042252 | RMSE |
| original_station_input | annual | overall | Matern | 2.678401 | RMSE |
| original_station_input | annual | overall | RBF | 3.232119 | RMSE |
| original_station_input | annual | overall_standardized | Matern | 2.010107 | standardized_RMSE |
| original_station_input | annual | overall_standardized | RBF | 2.034542 | standardized_RMSE |
| original_station_input | monthly | mu | Matern | 1.722465 | RMSE |
| original_station_input | monthly | mu | RBF | 2.539293 | RMSE |
| original_station_input | monthly | sigma | Matern | 3.063145 | RMSE |
| original_station_input | monthly | sigma | RBF | 3.176291 | RMSE |
| original_station_input | monthly | xi | Matern | 0.025865 | RMSE |
| original_station_input | monthly | xi | RBF | 0.030023 | RMSE |
| original_station_input | monthly | overall | Matern | 2.087473 | RMSE |
| original_station_input | monthly | overall | RBF | 2.348448 | RMSE |
| original_station_input | monthly | overall_standardized | Matern | 1.533310 | standardized_RMSE |
| original_station_input | monthly | overall_standardized | RBF | 1.658949 | standardized_RMSE |
| gridded_input | annual_45 | mu | Matern | 0.072235 | standardized_RMSE |
| gridded_input | annual_45 | mu | RBF | 0.068455 | standardized_RMSE |
| gridded_input | annual_45 | sigma | Matern | 2.678828 | standardized_RMSE |
| gridded_input | annual_45 | sigma | RBF | 2.687382 | standardized_RMSE |
| gridded_input | annual_45 | xi | Matern | 4.256002 | standardized_RMSE |
| gridded_input | annual_45 | xi | RBF | 3.861606 | standardized_RMSE |
| gridded_input | annual_45 | overall | Matern | 0.250196 | RMSE |
| gridded_input | annual_45 | overall | RBF | 0.246821 | RMSE |
| gridded_input | annual_45 | overall_standardized | Matern | 2.904138 | standardized_RMSE |
| gridded_input | annual_45 | overall_standardized | RBF | 2.716536 | standardized_RMSE |
| gridded_input | monthly_540 | mu | Matern | 0.025950 | standardized_RMSE |
| gridded_input | monthly_540 | mu | RBF | 0.019273 | standardized_RMSE |
| gridded_input | monthly_540 | sigma | Matern | 1.442049 | standardized_RMSE |
| gridded_input | monthly_540 | sigma | RBF | 1.425994 | standardized_RMSE |
| gridded_input | monthly_540 | xi | Matern | 1.030043 | standardized_RMSE |
| gridded_input | monthly_540 | xi | RBF | 1.051729 | standardized_RMSE |
| gridded_input | monthly_540 | overall | Matern | 0.124881 | RMSE |
| gridded_input | monthly_540 | overall | RBF | 0.121876 | RMSE |
| gridded_input | monthly_540 | overall_standardized | Matern | 1.024811 | standardized_RMSE |
| gridded_input | monthly_540 | overall_standardized | RBF | 1.023062 | standardized_RMSE |
