# benchmark_500 Deposit Candidate

## Contents

`benchmark_500.csv` contains 500 curated medical
`(origin, relationship, target)` triples used in the JMIR AI manuscript:

> A Two-Stage Knowledge Graph and Large Language Model Pipeline for Safety
> Screening of Medical Facts: Development and Benchmark Evaluation Study

Candidate dataset version: `v1.0.0`.

## Columns

- `origin`: subject concept of the medical triple
- `relationship`: one of the 7 evaluated relation types
- `target`: object concept of the medical triple
- `ground_truth`: refined label (`correct`, `incorrect`, or `edge`)
- `source`: candidate-generation provenance label
- `domain`: clinical-domain label
- `rationale`: construction-stage rationale

## Label distribution and evaluation use

The refined label distribution is 171 `correct`, 158 `incorrect`, and 171
`edge`. Binary performance uses `correct` and `incorrect` only (n=329);
clinically conditional `edge` cases are excluded from the binary confusion
matrix and reported separately.

Ground-truth refinement used targeted disagreement review rather than a blinded
full-benchmark audit. The manuscript and Multimedia Appendices 4–5 report the
selection procedure, its potential bias, and all 11 reclassifications.

## Integrity and provenance

This file is copied unchanged from the protected canonical dataset in the
repository. Do not relabel or recalculate it as part of release packaging.

## Deposit status

The dataset is licensed under CC BY-NC 4.0; see
`LICENSE_CC_BY_NC_4.0.md`. The public DOI remains pending. Add the DOI,
publication date, and preferred citation before publishing the deposit.
