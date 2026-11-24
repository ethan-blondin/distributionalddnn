# Benchmark — run `neural_w7`
Test: 554 days per horizon. DM at 5 %: `DM_wins` = number of models beaten significantly (multivariate DM, Ziel & Weron 2018).
rMAE = MAE / MAE(naive). `naive` = similar-day naive of the paper (Y_{T-7} Mon/Sat/Sun, Y_{T-1} otherwise); `rMAE_weekly` uses Y_{T-7} for every day.

## 2015-2020

### Point forecasts (ranked by rMAE)

|                  |   MAE |   rMAE |   rMAE_weekly |   RMSE |   DM_wins |   rank_rMAE |   rank_rMAE_weekly |
|:-----------------|------:|-------:|--------------:|-------:|----------:|------------:|-------------------:|
| DDNN-JSU-pEns    | 3.597 |  0.409 |         0.371 |  7.457 |        18 |           1 |                  1 |
| DDNN-JSU-qEns    | 3.636 |  0.413 |         0.375 |  7.457 |        13 |           2 |                  2 |
| DDNN-Normal-pEns | 3.699 |  0.42  |         0.381 |  6.029 |        14 |           3 |                  3 |
| DDNN-Normal-qEns | 3.707 |  0.421 |         0.382 |  6.029 |        13 |           4 |                  4 |
| DNN-Ens          | 3.709 |  0.421 |         0.382 |  5.964 |        13 |           5 |                  5 |
| DNN-QRA          | 3.715 |  0.422 |         0.383 |  5.884 |        13 |           6 |                  6 |
| DNN-1            | 3.784 |  0.43  |         0.39  |  6.041 |        10 |           7 |                  7 |
| DDNN-Normal-1    | 3.825 |  0.434 |         0.394 |  6.199 |         9 |           8 |                  8 |
| DNN-2            | 3.874 |  0.44  |         0.399 |  6.153 |         8 |           9 |                  9 |
| DDNN-JSU-1       | 3.896 |  0.443 |         0.402 | 10.443 |         7 |          10 |                 10 |
| DDNN-Normal-2    | 3.938 |  0.447 |         0.406 |  6.211 |         7 |          11 |                 11 |
| DDNN-JSU-2       | 4.013 |  0.456 |         0.414 |  6.427 |         6 |          12 |                 12 |
| LEAR-QRA         | 4.112 |  0.467 |         0.424 |  6.717 |         6 |          13 |                 13 |
| LEAR-Ens         | 4.304 |  0.489 |         0.444 |  6.728 |         5 |          14 |                 14 |
| LEAR-1456        | 4.591 |  0.522 |         0.473 |  6.82  |         4 |          15 |                 15 |
| LEAR-1092        | 4.697 |  0.534 |         0.484 |  6.911 |         3 |          16 |                 16 |
| LEAR-56          | 5.214 |  0.592 |         0.538 |  8.043 |         1 |          17 |                 17 |
| LEAR-84          | 5.288 |  0.601 |         0.545 |  7.978 |         1 |          18 |                 18 |
| naive            | 8.802 |  1     |         0.907 | 13.683 |         0 |          19 |                 19 |

### Probabilistic forecasts (ranked by CRPS)

Central intervals 50 % (q25-q75), 90 % (q05-q95), 98 % (q01-q99). Per hour (of 24): `kupiec_*_ok` = unconditional coverage not rejected; `indep_*_ok` = Christoffersen independence not rejected; `christ_*_ok` = Christoffersen conditional coverage (chi2(2)) not rejected. `width_*` = sharpness (EUR/MWh, lower is sharper).

|                  |   CRPS |   rel_CRPS_vs_naive_% |   cover_50 |   width_50 |   kupiec_50_ok |   indep_50_ok |   christ_50_ok |   cover_90 |   width_90 |   kupiec_90_ok |   indep_90_ok |   christ_90_ok |   cover_98 |   width_98 |   kupiec_98_ok |   indep_98_ok |   christ_98_ok |   DM_wins |
|:-----------------|-------:|----------------------:|-----------:|-----------:|---------------:|--------------:|---------------:|-----------:|-----------:|---------------:|--------------:|---------------:|-----------:|-----------:|---------------:|--------------:|---------------:|----------:|
| DDNN-JSU-pEns    |  1.324 |                61.139 |     57.243 |      6.912 |              6 |            17 |              6 |     93.735 |     17.558 |              3 |            19 |              8 |     98.744 |     26.223 |             18 |            24 |             22 |         9 |
| DDNN-JSU-qEns    |  1.325 |                61.102 |     53.851 |      6.424 |             16 |            16 |             13 |     92.223 |     16.239 |             13 |            16 |             11 |     98.21  |     23.989 |             22 |            22 |             22 |         7 |
| DDNN-Normal-pEns |  1.363 |                59.984 |     55.739 |      6.945 |              6 |            15 |              5 |     92.178 |     17.157 |             15 |            17 |             11 |     97.796 |     24.525 |             21 |            22 |             21 |         6 |
| DDNN-Normal-qEns |  1.363 |                59.979 |     53.836 |      6.716 |             17 |            18 |             14 |     90.945 |     16.378 |             19 |            16 |             17 |     97.15  |     23.163 |             16 |            24 |             18 |         6 |
| DNN-QRA          |  1.413 |                58.515 |     48.007 |      4.898 |              4 |             9 |              1 |     88.696 |     15.692 |              6 |             7 |              3 |     97.315 |     30.903 |              9 |            19 |              9 |         3 |
| DDNN-Normal-1    |  1.414 |                58.484 |     52.489 |      6.723 |             16 |            20 |             14 |     88.636 |     16.395 |             19 |            13 |             14 |     96.217 |     23.187 |              9 |            19 |              8 |         3 |
| DDNN-JSU-1       |  1.42  |                58.329 |     50.632 |      6.488 |             21 |            20 |             17 |     89.576 |     16.347 |             16 |            15 |             10 |     97.112 |     24.077 |             15 |            21 |             14 |         2 |
| DDNN-Normal-2    |  1.439 |                57.743 |     50.038 |      6.709 |             23 |            16 |             19 |     88.41  |     16.361 |             18 |            17 |             15 |     96.051 |     23.14  |              7 |            21 |              7 |         2 |
| DDNN-JSU-2       |  1.459 |                57.168 |     49.007 |      6.359 |             23 |            17 |             17 |     88.576 |     16.131 |             16 |            12 |             10 |     97.029 |     23.9   |             16 |            20 |             14 |         2 |
| LEAR-QRA         |  1.558 |                54.276 |     48.669 |      5.578 |              6 |            17 |              6 |     88.124 |     16.604 |              2 |            10 |              1 |     97.195 |     34.121 |              7 |            18 |              6 |         1 |
| naive            |  3.407 |                 0     |     48.955 |     10.835 |             19 |             5 |              5 |     91.591 |     44.86  |             15 |             1 |              0 |     98.616 |     89.78  |             13 |            12 |              9 |         0 |

### Errors in the tails

`twCRPS_*`: quantile-weighted CRPS (Gneiting & Ranjan, 2011), weights (2τ-1)² / (1-τ)² / τ² normalised to mean 1. `pinball_q01-05` / `q95-99`: mean pinball loss of the 5 lowest / highest percentiles. `*_low5%` / `*_high5%`: hours with price ≤ 4.28 / ≥ 55.36 EUR/MWh (5 % / 95 % percentiles of the test prices); `*_neg`: the 369 hours with negative prices.

|                  |   twCRPS_both |   twCRPS_left |   twCRPS_right |   pinball_q01-05 |   pinball_q95-99 |   MAE_low5% |   CRPS_low5% |   MAE_high5% |   CRPS_high5% |   MAE_neg |   CRPS_neg |
|:-----------------|--------------:|--------------:|---------------:|-----------------:|-----------------:|------------:|-------------:|-------------:|--------------:|----------:|-----------:|
| DDNN-JSU-qEns    |         0.901 |         1.236 |          1.205 |            0.361 |            0.33  |      10.29  |        3.826 |        9.684 |         3.58  |    12.556 |      4.743 |
| DDNN-JSU-pEns    |         0.909 |         1.234 |          1.209 |            0.37  |            0.338 |      10.237 |        3.846 |        9.272 |         3.46  |    12.436 |      4.766 |
| DDNN-Normal-qEns |         0.936 |         1.254 |          1.262 |            0.351 |            0.387 |      10.037 |        3.884 |        9.282 |         3.512 |    13.025 |      5.044 |
| DDNN-Normal-pEns |         0.937 |         1.254 |          1.263 |            0.352 |            0.386 |      10.075 |        3.885 |        9.276 |         3.499 |    13.046 |      5.04  |
| DDNN-JSU-1       |         0.967 |         1.331 |          1.285 |            0.392 |            0.363 |      10.548 |        3.987 |       11.121 |         4.21  |    13.363 |      5.094 |
| DDNN-Normal-1    |         0.977 |         1.3   |          1.313 |            0.38  |            0.405 |      10.622 |        4.124 |        8.955 |         3.408 |    13.71  |      5.341 |
| DDNN-Normal-2    |         0.984 |         1.328 |          1.327 |            0.369 |            0.41  |       9.863 |        3.774 |        9.888 |         3.756 |    12.853 |      4.88  |
| DDNN-JSU-2       |         0.99  |         1.359 |          1.329 |            0.393 |            0.367 |      12.021 |        4.371 |        8.929 |         3.319 |    14.736 |      5.376 |
| DNN-QRA          |         1.023 |         1.332 |          1.302 |            0.513 |            0.45  |      10.034 |        4.081 |        8.789 |         3.658 |    12.679 |      5.282 |
| LEAR-QRA         |         1.118 |         1.498 |          1.401 |            0.596 |            0.433 |      14.579 |        5.862 |        9.434 |         3.754 |    19.076 |      7.889 |
| naive            |         2.491 |         3.209 |          3.153 |            1.183 |            1.106 |      27.945 |       11.596 |       13.118 |         4.965 |    33.637 |     14.196 |

### Diebold-Mariano variants

`wins_*` = number of models beaten at 5 %. L1: multivariate, daily sum of the 24 hourly losses (Ziel & Weron, 2018, as in the pipeline). L2: daily Euclidean norm. L1-NW: L1 with Newey-West variance (lag 7) and Harvey-Leybourne-Newbold t(n-1) reference. `hours_better_total`: univariate DM per hour, number of (hour, opponent) pairs where the model is significantly better. Full p-value matrices in `benchmark_dm_*.csv`.

Probabilistic (loss = hourly CRPS):

| column (better?)   |   wins_L1 |   wins_L2 |   wins_L1-NW |   hours_better_total |
|:-------------------|----------:|----------:|-------------:|---------------------:|
| naive              |         0 |         0 |            0 |                    0 |
| LEAR-QRA           |         1 |         1 |            1 |                   26 |
| DNN-QRA            |         3 |         2 |            2 |                   55 |
| DDNN-JSU-1         |         2 |         1 |            2 |                   59 |
| DDNN-JSU-2         |         2 |         2 |            2 |                   35 |
| DDNN-JSU-pEns      |         9 |         8 |            9 |                  157 |
| DDNN-JSU-qEns      |         7 |         7 |            7 |                  157 |
| DDNN-Normal-1      |         3 |         2 |            3 |                   56 |
| DDNN-Normal-2      |         2 |         2 |            2 |                   44 |
| DDNN-Normal-pEns   |         6 |         7 |            6 |                  121 |
| DDNN-Normal-qEns   |         6 |         6 |            6 |                  124 |

Point (loss = absolute error of the median):

| column (better?)   |   wins_L1 |   wins_L2 |   wins_L1-NW |   hours_better_total |
|:-------------------|----------:|----------:|-------------:|---------------------:|
| DDNN-JSU-pEns      |        18 |        14 |           18 |                  296 |
| DDNN-JSU-qEns      |        13 |        10 |           13 |                  267 |
| DDNN-Normal-pEns   |        14 |        12 |           14 |                  250 |
| DDNN-Normal-qEns   |        13 |        11 |           12 |                  237 |
| DNN-Ens            |        13 |        12 |           13 |                  230 |
| DNN-QRA            |        13 |        12 |           13 |                  224 |
| DNN-1              |        10 |        10 |           10 |                  178 |
| DDNN-Normal-1      |         9 |         8 |            8 |                  167 |
| DNN-2              |         8 |         8 |            8 |                  150 |
| DDNN-JSU-1         |         7 |         5 |            7 |                  145 |
| DDNN-Normal-2      |         7 |         7 |            7 |                  135 |
| DDNN-JSU-2         |         6 |         6 |            6 |                  127 |
| LEAR-QRA           |         6 |         6 |            6 |                  148 |
| LEAR-Ens           |         5 |         5 |            5 |                  109 |
| LEAR-1456          |         4 |         4 |            4 |                   74 |
| LEAR-1092          |         3 |         3 |            3 |                   57 |
| LEAR-56            |         1 |         1 |            1 |                   32 |
| LEAR-84            |         1 |         1 |            1 |                   24 |
| naive              |         0 |         0 |            0 |                    0 |

## 2020-2026

### Point forecasts (ranked by rMAE)

|                  |    MAE |   rMAE |   rMAE_weekly |   RMSE |   DM_wins |   rank_rMAE |   rank_rMAE_weekly |
|:-----------------|-------:|-------:|--------------:|-------:|----------:|------------:|-------------------:|
| DNN-QRA          | 14.464 |  0.436 |         0.375 | 24.602 |        17 |           1 |                  1 |
| LEAR-QRA         | 14.721 |  0.444 |         0.381 | 24.695 |        17 |           2 |                  2 |
| DDNN-JSU-pEns    | 15.261 |  0.46  |         0.395 | 24.604 |        10 |           3 |                  3 |
| DDNN-JSU-qEns    | 15.291 |  0.461 |         0.396 | 24.604 |        10 |           4 |                  4 |
| DNN-Ens          | 15.336 |  0.462 |         0.397 | 25.447 |        10 |           5 |                  5 |
| LEAR-Ens         | 15.372 |  0.463 |         0.398 | 25.088 |         9 |           6 |                  6 |
| DNN-2            | 15.479 |  0.467 |         0.401 | 25.509 |         9 |           7 |                  7 |
| DDNN-Normal-pEns | 15.513 |  0.468 |         0.402 | 25.326 |         9 |           8 |                  8 |
| DDNN-Normal-qEns | 15.52  |  0.468 |         0.402 | 25.326 |         9 |           9 |                  9 |
| DDNN-JSU-1       | 15.587 |  0.47  |         0.404 | 25.861 |         7 |          10 |                 10 |
| DDNN-Normal-1    | 15.846 |  0.478 |         0.411 | 25.405 |         5 |          11 |                 11 |
| DNN-1            | 15.936 |  0.48  |         0.413 | 26.136 |         5 |          12 |                 12 |
| DDNN-Normal-2    | 16.031 |  0.483 |         0.415 | 26.153 |         5 |          13 |                 13 |
| LEAR-1092        | 16.185 |  0.488 |         0.419 | 26.189 |         5 |          14 |                 14 |
| DDNN-JSU-2       | 16.879 |  0.509 |         0.437 | 26.244 |         3 |          15 |                 15 |
| LEAR-1456        | 16.917 |  0.51  |         0.438 | 26.388 |         3 |          16 |                 16 |
| LEAR-56          | 18.304 |  0.552 |         0.474 | 28.705 |         2 |          17 |                 17 |
| LEAR-84          | 18.707 |  0.564 |         0.485 | 28.992 |         1 |          18 |                 18 |
| naive            | 33.173 |  1     |         0.859 | 52.913 |         0 |          19 |                 19 |

### Probabilistic forecasts (ranked by CRPS)

Central intervals 50 % (q25-q75), 90 % (q05-q95), 98 % (q01-q99). Per hour (of 24): `kupiec_*_ok` = unconditional coverage not rejected; `indep_*_ok` = Christoffersen independence not rejected; `christ_*_ok` = Christoffersen conditional coverage (chi2(2)) not rejected. `width_*` = sharpness (EUR/MWh, lower is sharper).

|                  |   CRPS |   rel_CRPS_vs_naive_% |   cover_50 |   width_50 |   kupiec_50_ok |   indep_50_ok |   christ_50_ok |   cover_90 |   width_90 |   kupiec_90_ok |   indep_90_ok |   christ_90_ok |   cover_98 |   width_98 |   kupiec_98_ok |   indep_98_ok |   christ_98_ok |   DM_wins |
|:-----------------|-------:|----------------------:|-----------:|-----------:|---------------:|--------------:|---------------:|-----------:|-----------:|---------------:|--------------:|---------------:|-----------:|-----------:|---------------:|--------------:|---------------:|----------:|
| DNN-QRA          |  5.475 |                58.552 |     44.141 |     17.698 |              9 |             5 |              2 |     87.049 |     53.493 |              3 |            10 |              2 |     96.969 |    116.682 |              9 |            15 |              7 |         7 |
| DDNN-JSU-qEns    |  5.565 |                57.866 |     49.195 |     24.517 |             20 |             9 |             10 |     89.568 |     61.493 |             17 |             5 |              5 |     97.503 |     89.969 |             22 |            16 |             15 |         5 |
| DDNN-JSU-pEns    |  5.566 |                57.861 |     51.737 |     25.962 |             18 |             7 |              5 |     91.704 |     66.01  |             16 |             3 |              5 |     98.172 |     98.044 |             22 |            18 |             20 |         5 |
| LEAR-QRA         |  5.607 |                57.552 |     45.615 |     17.821 |              3 |            14 |              1 |     87.831 |     56.704 |              1 |             8 |              0 |     97.315 |    115.572 |             11 |            11 |             12 |         4 |
| DDNN-Normal-qEns |  5.665 |                57.11  |     49.692 |     26.099 |             14 |             7 |              5 |     89.922 |     63.646 |             15 |            10 |              8 |     97.225 |     90.016 |             19 |            16 |             16 |         4 |
| DDNN-Normal-pEns |  5.666 |                57.103 |     50.752 |     26.56  |             14 |             7 |              3 |     91.005 |     65.891 |             18 |            11 |             10 |     97.774 |     94.435 |             24 |            17 |             19 |         4 |
| DDNN-JSU-1       |  5.714 |                56.743 |     49.21  |     24.332 |             23 |             4 |              8 |     88.154 |     60.459 |             17 |             4 |              4 |     95.939 |     87.516 |              4 |             8 |              2 |         3 |
| DDNN-Normal-1    |  5.797 |                56.117 |     48.353 |     26.104 |             13 |             6 |              4 |     87.5   |     63.659 |             13 |             8 |              8 |     95.984 |     90.034 |             10 |            12 |             10 |         2 |
| DDNN-Normal-2    |  5.886 |                55.441 |     48.308 |     26.094 |             14 |             8 |              8 |     88.177 |     63.634 |             13 |             6 |              6 |     96.382 |     89.999 |             11 |            17 |             10 |         2 |
| DDNN-JSU-2       |  6.185 |                53.175 |     44.991 |     24.701 |              9 |            10 |              6 |     84.589 |     62.526 |              6 |             3 |              2 |     95.104 |     92.422 |              6 |             9 |              5 |         1 |
| naive            | 13.209 |                 0     |     52.971 |     40.897 |             15 |             0 |              1 |     89.952 |    171.135 |             21 |             0 |              0 |     97.21  |    324.8   |             18 |             1 |              2 |         0 |

### Errors in the tails

`twCRPS_*`: quantile-weighted CRPS (Gneiting & Ranjan, 2011), weights (2τ-1)² / (1-τ)² / τ² normalised to mean 1. `pinball_q01-05` / `q95-99`: mean pinball loss of the 5 lowest / highest percentiles. `*_low5%` / `*_high5%`: hours with price ≤ -0.09 / ≥ 224.99 EUR/MWh (5 % / 95 % percentiles of the test prices); `*_neg`: the 856 hours with negative prices.

|                  |   twCRPS_both |   twCRPS_left |   twCRPS_right |   pinball_q01-05 |   pinball_q95-99 |   MAE_low5% |   CRPS_low5% |   MAE_high5% |   CRPS_high5% |   MAE_neg |   CRPS_neg |
|:-----------------|--------------:|--------------:|---------------:|-----------------:|-----------------:|------------:|-------------:|-------------:|--------------:|----------:|-----------:|
| DDNN-JSU-qEns    |         3.797 |         5.041 |          5.219 |            1.377 |            1.666 |      21.011 |        7.963 |       44.87  |        17.419 |    19.537 |      7.343 |
| DDNN-JSU-pEns    |         3.807 |         5.08  |          5.186 |            1.413 |            1.65  |      19.785 |        7.705 |       45.267 |        17.512 |    18.461 |      7.119 |
| DDNN-Normal-qEns |         3.886 |         5.178 |          5.276 |            1.477 |            1.715 |      20.601 |        7.767 |       44.994 |        18.16  |    19.045 |      7.13  |
| DDNN-Normal-pEns |         3.891 |         5.194 |          5.264 |            1.499 |            1.7   |      20.524 |        7.767 |       45.043 |        18.186 |    18.989 |      7.134 |
| DDNN-JSU-1       |         3.939 |         5.14  |          5.413 |            1.417 |            1.875 |      20.776 |        7.884 |       48.065 |        18.766 |    19.064 |      7.165 |
| DNN-QRA          |         3.942 |         4.923 |          5.272 |            1.56  |            2.175 |      16.856 |        6.871 |       44.976 |        19.239 |    15.61  |      6.242 |
| DDNN-Normal-1    |         3.985 |         5.273 |          5.428 |            1.506 |            1.785 |      21.344 |        7.999 |       45.624 |        18.263 |    19.614 |      7.313 |
| LEAR-QRA         |         4.048 |         5.133 |          5.313 |            1.663 |            2.121 |      20.1   |        7.917 |       46.009 |        19.662 |    19.065 |      7.348 |
| DDNN-Normal-2    |         4.057 |         5.336 |          5.535 |            1.503 |            1.868 |      20.961 |        7.895 |       46.269 |        18.743 |    19.461 |      7.3   |
| DDNN-JSU-2       |         4.256 |         5.512 |          5.908 |            1.518 |            1.953 |      24.257 |        9.122 |       46.31  |        18.003 |    22.851 |      8.536 |
| naive            |         9.909 |        12.386 |         12.406 |            4.618 |            4.706 |      39.656 |       16.41  |       84.911 |        35.087 |    37.689 |     15.696 |

### Diebold-Mariano variants

`wins_*` = number of models beaten at 5 %. L1: multivariate, daily sum of the 24 hourly losses (Ziel & Weron, 2018, as in the pipeline). L2: daily Euclidean norm. L1-NW: L1 with Newey-West variance (lag 7) and Harvey-Leybourne-Newbold t(n-1) reference. `hours_better_total`: univariate DM per hour, number of (hour, opponent) pairs where the model is significantly better. Full p-value matrices in `benchmark_dm_*.csv`.

Probabilistic (loss = hourly CRPS):

| column (better?)   |   wins_L1 |   wins_L2 |   wins_L1-NW |   hours_better_total |
|:-------------------|----------:|----------:|-------------:|---------------------:|
| naive              |         0 |         0 |            0 |                    0 |
| LEAR-QRA           |         4 |         1 |            3 |                  110 |
| DNN-QRA            |         7 |         3 |            4 |                  132 |
| DDNN-JSU-1         |         3 |         3 |            2 |                   59 |
| DDNN-JSU-2         |         1 |         1 |            1 |                   25 |
| DDNN-JSU-pEns      |         5 |        10 |            5 |                  114 |
| DDNN-JSU-qEns      |         5 |         9 |            5 |                  113 |
| DDNN-Normal-1      |         2 |         4 |            2 |                   52 |
| DDNN-Normal-2      |         2 |         2 |            2 |                   40 |
| DDNN-Normal-pEns   |         4 |         6 |            4 |                   97 |
| DDNN-Normal-qEns   |         4 |         5 |            4 |                   94 |

Point (loss = absolute error of the median):

| column (better?)   |   wins_L1 |   wins_L2 |   wins_L1-NW |   hours_better_total |
|:-------------------|----------:|----------:|-------------:|---------------------:|
| DNN-QRA            |        17 |        18 |           17 |                  333 |
| LEAR-QRA           |        17 |        11 |           14 |                  270 |
| DDNN-JSU-pEns      |        10 |        15 |            9 |                  203 |
| DDNN-JSU-qEns      |        10 |        13 |            9 |                  202 |
| DNN-Ens            |        10 |         9 |            9 |                  209 |
| LEAR-Ens           |         9 |         6 |            6 |                  221 |
| DNN-2              |         9 |         9 |            9 |                  177 |
| DDNN-Normal-pEns   |         9 |         9 |            8 |                  193 |
| DDNN-Normal-qEns   |         9 |         9 |            8 |                  188 |
| DDNN-JSU-1         |         7 |         6 |            5 |                  146 |
| DDNN-Normal-1      |         5 |         6 |            5 |                  122 |
| DNN-1              |         5 |         6 |            5 |                  123 |
| DDNN-Normal-2      |         5 |         6 |            4 |                  109 |
| LEAR-1092          |         5 |         4 |            4 |                  167 |
| DDNN-JSU-2         |         3 |         3 |            3 |                   75 |
| LEAR-1456          |         3 |         3 |            3 |                  119 |
| LEAR-56            |         2 |         2 |            2 |                   35 |
| LEAR-84            |         1 |         1 |            1 |                   27 |
| naive              |         0 |         0 |            0 |                    0 |

## Cross-period comparison (scale-free metrics)

|                  |   rMAE 2015-2020 |   rMAE 2020-2026 |   rel CRPS % 2015-2020 |   rel CRPS % 2020-2026 |
|:-----------------|-----------------:|-----------------:|-----------------------:|-----------------------:|
| DDNN-JSU-pEns    |            0.409 |            0.46  |                 61.139 |                 57.861 |
| DDNN-JSU-qEns    |            0.413 |            0.461 |                 61.102 |                 57.866 |
| DDNN-Normal-pEns |            0.42  |            0.468 |                 59.984 |                 57.103 |
| DDNN-Normal-qEns |            0.421 |            0.468 |                 59.979 |                 57.11  |
| DNN-Ens          |            0.421 |            0.462 |                nan     |                nan     |
| DNN-QRA          |            0.422 |            0.436 |                 58.515 |                 58.552 |
| DNN-1            |            0.43  |            0.48  |                nan     |                nan     |
| DDNN-Normal-1    |            0.434 |            0.478 |                 58.484 |                 56.117 |
| DNN-2            |            0.44  |            0.467 |                nan     |                nan     |
| DDNN-JSU-1       |            0.443 |            0.47  |                 58.329 |                 56.743 |
| DDNN-Normal-2    |            0.447 |            0.483 |                 57.743 |                 55.441 |
| DDNN-JSU-2       |            0.456 |            0.509 |                 57.168 |                 53.175 |
| LEAR-QRA         |            0.467 |            0.444 |                 54.276 |                 57.552 |
| LEAR-Ens         |            0.489 |            0.463 |                nan     |                nan     |
| LEAR-1456        |            0.522 |            0.51  |                nan     |                nan     |
| LEAR-1092        |            0.534 |            0.488 |                nan     |                nan     |
| LEAR-56          |            0.592 |            0.552 |                nan     |                nan     |
| LEAR-84          |            0.601 |            0.564 |                nan     |                nan     |
| naive            |            1     |            1     |                  0     |                  0     |

## Computation time (single-worker hours, summed over fits)

NN: per-fit timer (machine of the NN run). LEAR: run log of the LEAR run. Wall time = total / number of parallel workers. Caveat: a few NN fits last several hours (machine asleep or overloaded), e.g. DNN-1 2020-2026 has 4 fits > 1 h; medians are more reliable.

|               |   2015-2020 |   2020-2026 |   total_h |
|:--------------|------------:|------------:|----------:|
| DDNN-JSU-1    |        3.66 |        5.34 |      9    |
| DDNN-JSU-2    |        1.81 |        3.08 |      4.89 |
| DDNN-Normal-1 |        3.43 |        7.07 |     10.5  |
| DDNN-Normal-2 |        1.42 |        3.03 |      4.45 |
| DNN-1         |       11.07 |       45.87 |     56.94 |
| DNN-2         |        6.33 |        6.97 |     13.3  |
| LEAR-1092     |        3.92 |        3.11 |      7.03 |
| LEAR-1456     |        5.51 |        4.46 |      9.97 |
| LEAR-56       |        0.09 |        0.07 |      0.16 |
| LEAR-84       |        0.12 |        0.12 |      0.24 |
| LEAR-QRA      |        0.45 |        0.47 |      0.92 |
