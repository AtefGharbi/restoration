## Primary contrast: Delta = ENS(B8) - ENS(B7), kWh/episode, Scenarios 2-3 equally weighted

Negative values favour B8. Two-sided 97.5% Welch intervals (Bonferroni over two feeders).

| Feeder | mean B8 | mean B7 | Delta | SE | 97.5% CI | CI width | Hedges g | df | episode-weighted Delta [97.5% CI] | crossed-bootstrap 97.5% | rand. p (Holm, exploratory) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee33 | 918.5 | 955.1 | -36.6 | 27.2 | [-103.3, 30.0] | 133.3 | -0.58 | 17.9 | -36.6 [-103.3, 30.0] | [-99.3, 20.1] | 0.196 |

ENS decomposition ieee33: outage component Delta -36.6 [-103.3, 30.0]; supervisor-shed component Delta 0.0 [0.0, 0.0].

## Secondary (descriptive): all methods, Scenarios 2-3, seed-level mean (sd over seeds)

| Feeder | Method | seeds | ENS_kWh | ENS_outage_kWh | ENS_shed_kWh | LRR | DA_completeness | CLR | n_switch | steps | success rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee33 | B0 | 1 | 1164.6 | 1164.6 | 0.0 | 0.000 | 0.671 | 0.000 | 0.0 | 2.9 | 0.05 |
| ieee33 | B1 | 1 | 858.6 | 858.6 | 0.0 | 0.413 | 0.797 | 0.376 | 1.4 | 3.5 | 0.32 |
| ieee33 | B3 | 1 | 852.6 | 852.6 | 0.0 | 0.412 | 0.797 | 0.376 | 1.4 | 3.4 | 0.32 |
| ieee33 | B7 | 10 | 955.1 (58.0) | 955.1 (58.0) | 0.0 (0.0) | 0.285 (0.056) | 0.755 (0.019) | 0.256 (0.079) | 2.1 (1.0) | 3.3 (0.3) | 0.19 |
| ieee33 | B8 | 10 | 918.5 (63.6) | 918.5 (63.6) | 0.0 (0.0) | 0.327 (0.086) | 0.772 (0.030) | 0.300 (0.076) | 2.5 (1.3) | 3.4 (0.3) | 0.24 |

## Other ladder rungs (secondary, 95% Welch, no confirmatory claim)

| Contrast | Delta ENS | 95% CI |
|---|---|---|

## B4 optimization reference (test instances, Scenarios 2-3)

| Feeder | J^ref mean | B4 served / P^DA | B4 Hamming | P^DA solves not proven optimal | max P^DA time (s) |
|---|---|---|---|---|---|
| ieee33 | 0.997 | 0.949 | 2.91 | 7 of 355 | 7.4 |

## Dependency-graph edges per test episode (B8, seed 0)

| Feeder | Scenario | E_P | E_R | E_C | E_E | unrestorable faults | episodes with >=1 edge |
|---|---|---|---|---|---|---|---|
| ieee33 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 |
| ieee33 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 0.00 |
| ieee33 | 3 | 0.00 | 0.00 | 0.12 | 0.12 | 2.70 | 0.14 |

## Measured wall-clock components (seconds per test episode, mean over seeds)

| Feeder | Method | supervisor | local_masks | actor_forward | assembly | env_step | G_F_update | baseline_policy | train wall (s/run) |
|---|---|---|---|---|---|---|---|---|---|
| ieee33 | B0 | 0.42 ms | - | - | - | 0.11 ms | - | 0.00 ms | nan |
| ieee33 | B1 | 0.29 ms | - | - | - | 0.17 ms | 25.57 ms | 3.53 ms | nan |
| ieee33 | B3 | 0.24 ms | - | - | - | 0.08 ms | 42.78 ms | 0.56 ms | nan |
| ieee33 | B7 | 0.41 ms | 5.72 ms | 3.49 ms | 0.15 ms | 0.21 ms | - | - | 53 |
| ieee33 | B8 | 0.22 ms | 2.51 ms | 1.93 ms | 0.10 ms | 0.10 ms | 22.52 ms | - | 60 |