# Benchmark — run `neural_w7`
Test: 554 days per horizon. DM at 5 %: `DM_wins` = number of models beaten significantly (multivariate DM, Ziel & Weron 2018).
rMAE = MAE / MAE(naive). `naive` = similar-day naive of the paper (Y_{T-7} Mon/Sat/Sun, Y_{T-1} otherwise); `rMAE_weekly` uses Y_{T-7} for every day.

## 2015-2020

### Point forecasts (ranked by rMAE)

|               |   MAE |   rMAE |   rMAE_weekly |   RMSE |   DM_wins |   rank_rMAE |   rank_rMAE_weekly |
|:--------------|------:|-------:|--------------:|-------:|----------:|------------:|-------------------:|
| DDNN-JSU-pEns | 3.597 |  0.409 |         0.371 |  7.457 |        14 |           1 |                  1 |
| DDNN-JSU-qEns | 3.636 |  0.413 |         0.375 |  7.457 |        11 |           2 |                  2 |
| DNN-Ens       | 3.709 |  0.421 |         0.382 |  5.964 |        11 |           3 |                  3 |
| DNN-QRA       | 3.715 |  0.422 |         0.383 |  5.884 |        11 |           4 |                  4 |
| DNN-1         | 3.784 |  0.43  |         0.39  |  6.041 |         9 |           5 |                  5 |
| DNN-2         | 3.874 |  0.44  |         0.399 |  6.153 |         8 |           6 |                  6 |
| DDNN-JSU-1    | 3.896 |  0.443 |         0.402 | 10.443 |         7 |           7 |                  7 |
| DDNN-JSU-2    | 4.013 |  0.456 |         0.414 |  6.427 |         6 |           8 |                  8 |
| LEAR-QRA      | 4.112 |  0.467 |         0.424 |  6.717 |         6 |           9 |                  9 |
| LEAR-Ens      | 4.304 |  0.489 |         0.444 |  6.728 |         5 |          10 |                 10 |
| LEAR-1456     | 4.591 |  0.522 |         0.473 |  6.82  |         4 |          11 |                 11 |
| LEAR-1092     | 4.697 |  0.534 |         0.484 |  6.911 |         3 |          12 |                 12 |
| LEAR-56       | 5.214 |  0.592 |         0.538 |  8.043 |         1 |          13 |                 13 |
| LEAR-84       | 5.288 |  0.601 |         0.545 |  7.978 |         1 |          14 |                 14 |
| naive         | 8.802 |  1     |         0.907 | 13.683 |         0 |          15 |                 15 |

### Probabilistic forecasts (ranked by CRPS)

Nominal coverage: 50 % and 90 %; `kupiec_*_ok` = hours (of 24) where Kupiec does not reject; `width_*` = sharpness (EUR/MWh, lower is sharper).

|               |   CRPS |   rel_CRPS_vs_naive_% |   cover_50 |   width_50 |   kupiec_50_ok |   cover_90 |   width_90 |   kupiec_90_ok |   DM_wins |
|:--------------|-------:|----------------------:|-----------:|-----------:|---------------:|-----------:|-----------:|---------------:|----------:|
| DDNN-JSU-pEns |  1.324 |                61.139 |     57.243 |      6.912 |              6 |     93.735 |     17.558 |              3 |         5 |
| DDNN-JSU-qEns |  1.325 |                61.102 |     53.851 |      6.424 |             16 |     92.223 |     16.239 |             13 |         5 |
| DNN-QRA       |  1.413 |                58.515 |     48.007 |      4.898 |              4 |     88.696 |     15.692 |              6 |         3 |
| DDNN-JSU-1    |  1.42  |                58.329 |     50.632 |      6.488 |             21 |     89.576 |     16.347 |             16 |         2 |
| DDNN-JSU-2    |  1.459 |                57.168 |     49.007 |      6.359 |             23 |     88.576 |     16.131 |             16 |         2 |
| LEAR-QRA      |  1.558 |                54.276 |     48.669 |      5.578 |              6 |     88.124 |     16.604 |              2 |         1 |
| naive         |  3.407 |                 0     |     48.955 |     10.835 |             19 |     91.591 |     44.86  |             15 |         0 |

## 2020-2026

### Point forecasts (ranked by rMAE)

|               |    MAE |   rMAE |   rMAE_weekly |   RMSE |   DM_wins |   rank_rMAE |   rank_rMAE_weekly |
|:--------------|-------:|-------:|--------------:|-------:|----------:|------------:|-------------------:|
| DNN-QRA       | 14.464 |  0.436 |         0.375 | 24.602 |        13 |           1 |                  1 |
| LEAR-QRA      | 14.721 |  0.444 |         0.381 | 24.695 |        13 |           2 |                  2 |
| DDNN-JSU-pEns | 15.261 |  0.46  |         0.395 | 24.604 |         8 |           3 |                  3 |
| DDNN-JSU-qEns | 15.291 |  0.461 |         0.396 | 24.604 |         8 |           4 |                  4 |
| DNN-Ens       | 15.336 |  0.462 |         0.397 | 25.447 |         8 |           5 |                  5 |
| LEAR-Ens      | 15.372 |  0.463 |         0.398 | 25.088 |         7 |           6 |                  6 |
| DNN-2         | 15.479 |  0.467 |         0.401 | 25.509 |         7 |           7 |                  7 |
| DDNN-JSU-1    | 15.587 |  0.47  |         0.404 | 25.861 |         6 |           8 |                  8 |
| DNN-1         | 15.936 |  0.48  |         0.413 | 26.136 |         5 |           9 |                  9 |
| LEAR-1092     | 16.185 |  0.488 |         0.419 | 26.189 |         5 |          10 |                 10 |
| DDNN-JSU-2    | 16.879 |  0.509 |         0.437 | 26.244 |         3 |          11 |                 11 |
| LEAR-1456     | 16.917 |  0.51  |         0.438 | 26.388 |         3 |          12 |                 12 |
| LEAR-56       | 18.304 |  0.552 |         0.474 | 28.705 |         2 |          13 |                 13 |
| LEAR-84       | 18.707 |  0.564 |         0.485 | 28.992 |         1 |          14 |                 14 |
| naive         | 33.173 |  1     |         0.859 | 52.913 |         0 |          15 |                 15 |

### Probabilistic forecasts (ranked by CRPS)

Nominal coverage: 50 % and 90 %; `kupiec_*_ok` = hours (of 24) where Kupiec does not reject; `width_*` = sharpness (EUR/MWh, lower is sharper).

|               |   CRPS |   rel_CRPS_vs_naive_% |   cover_50 |   width_50 |   kupiec_50_ok |   cover_90 |   width_90 |   kupiec_90_ok |   DM_wins |
|:--------------|-------:|----------------------:|-----------:|-----------:|---------------:|-----------:|-----------:|---------------:|----------:|
| DNN-QRA       |  5.475 |                58.552 |     44.141 |     17.698 |              9 |     87.049 |     53.493 |              3 |         3 |
| DDNN-JSU-qEns |  5.565 |                57.866 |     49.195 |     24.517 |             20 |     89.568 |     61.493 |             17 |         3 |
| DDNN-JSU-pEns |  5.566 |                57.861 |     51.737 |     25.962 |             18 |     91.704 |     66.01  |             16 |         3 |
| LEAR-QRA      |  5.607 |                57.552 |     45.615 |     17.821 |              3 |     87.831 |     56.704 |              1 |         2 |
| DDNN-JSU-1    |  5.714 |                56.743 |     49.21  |     24.332 |             23 |     88.154 |     60.459 |             17 |         2 |
| DDNN-JSU-2    |  6.185 |                53.175 |     44.991 |     24.701 |              9 |     84.589 |     62.526 |              6 |         1 |
| naive         | 13.209 |                 0     |     52.971 |     40.897 |             15 |     89.952 |    171.135 |             21 |         0 |

## Cross-period comparison (scale-free metrics)

|               |   rMAE 2015-2020 |   rMAE 2020-2026 |   rel CRPS % 2015-2020 |   rel CRPS % 2020-2026 |
|:--------------|-----------------:|-----------------:|-----------------------:|-----------------------:|
| DDNN-JSU-pEns |            0.409 |            0.46  |                 61.139 |                 57.861 |
| DDNN-JSU-qEns |            0.413 |            0.461 |                 61.102 |                 57.866 |
| DNN-Ens       |            0.421 |            0.462 |                nan     |                nan     |
| DNN-QRA       |            0.422 |            0.436 |                 58.515 |                 58.552 |
| DNN-1         |            0.43  |            0.48  |                nan     |                nan     |
| DNN-2         |            0.44  |            0.467 |                nan     |                nan     |
| DDNN-JSU-1    |            0.443 |            0.47  |                 58.329 |                 56.743 |
| DDNN-JSU-2    |            0.456 |            0.509 |                 57.168 |                 53.175 |
| LEAR-QRA      |            0.467 |            0.444 |                 54.276 |                 57.552 |
| LEAR-Ens      |            0.489 |            0.463 |                nan     |                nan     |
| LEAR-1456     |            0.522 |            0.51  |                nan     |                nan     |
| LEAR-1092     |            0.534 |            0.488 |                nan     |                nan     |
| LEAR-56       |            0.592 |            0.552 |                nan     |                nan     |
| LEAR-84       |            0.601 |            0.564 |                nan     |                nan     |
| naive         |            1     |            1     |                  0     |                  0     |

## Computation time (single-worker hours, summed over fits)

NN: per-fit timer (machine of the NN run). LEAR: run log of the LEAR run. Wall time = total / number of parallel workers. Caveat: a few NN fits last several hours (machine asleep or overloaded), e.g. DNN-1 2020-2026 has 4 fits > 1 h; medians are more reliable.

|            |   2015-2020 |   2020-2026 |   total_h |
|:-----------|------------:|------------:|----------:|
| DDNN-JSU-1 |        3.66 |        5.34 |      9    |
| DDNN-JSU-2 |        1.81 |        3.08 |      4.89 |
| DNN-1      |       11.07 |       45.87 |     56.94 |
| DNN-2      |        6.33 |        6.97 |     13.3  |
| LEAR-1092  |        3.92 |        3.11 |      7.03 |
| LEAR-1456  |        5.51 |        4.46 |      9.97 |
| LEAR-56    |        0.09 |        0.07 |      0.16 |
| LEAR-84    |        0.12 |        0.12 |      0.24 |
| LEAR-QRA   |        0.45 |        0.47 |      0.92 |
