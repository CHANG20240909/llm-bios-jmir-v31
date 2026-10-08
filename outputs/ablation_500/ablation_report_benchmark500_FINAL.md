# Phase 3: Ablation Study Results

Ground truth: 500 triples (correct=171, incorrect=158, edge=171)

Reference: Alber et al. (Nature Medicine 2025) Triplet-level P=79.7%, R=81.3%, F1=80.5%


## Primary Metrics (correct + incorrect only)

| Condition | Precision | 95% CI | Recall | 95% CI | F1 | FPR | Miss Rate |
|-----------|:---------:|:------:|:------:|:------:|:--:|:---:|:---------:|
| C3 | 64.0% | [57.8%, 69.7%] | 100.0% | [97.6%, 100.0%] | 78.0% | 52.0% | 0.0% |
| C4 | 89.3% | [83.8%, 93.0%] | 100.0% | [97.6%, 100.0%] | 94.3% | 11.1% | 0.0% |

## Confusion Matrix

| Condition | TP | FN (missed!) | FP (false alarm) | TN |
|-----------|:--:|:------------:|:----------------:|:--:|
| C3 | 158 | 0 | 89 | 82 |
| C4 | 158 | 0 | 19 | 152 |

## Edge Case Analysis

| Condition | PASS | WARN | FAIL | Total |
|-----------|:----:|:----:|:----:|:-----:|
| C3 | 60 | 44 | 67 | 171 |
| C4 | 133 | 15 | 23 | 171 |

## Pairwise Comparison (McNemar's test)

| Pair | Discordant (b, c) | p-value | Significant? |
|------|:-----------------:|:-------:|:------------:|
| C3 vs C4 | (0, 70) | 0.0000 | Yes |
