# LLM-BIOS: JMIR AI v3.1 Reproducibility Snapshot

This repository is the sanitized public release candidate for the manuscript:

> A Two-Stage Knowledge Graph and Large Language Model Pipeline for Safety
> Screening of Medical Facts: Development and Benchmark Evaluation Study

It preserves the evaluated C4 configuration: Phase 1 C3 screening followed by
the GPT-4o Phase 2 verifier with the v3.1 prompt, G1–G4 guardrails, and
deterministic post-processing.

## Release scope

- The v3.1 `SYSTEM_PROMPT` in `nyuolab-src-clean/llm_agent.py` was restored
  byte-for-byte from commit `ec0791be6305f1062aabf0810a55caed4e08db5e`.
- The later unevaluated v3.2 prompt change and T3-A evidence-augmentation work
  are not part of this release candidate.
- Frozen benchmark and result artifacts are retained without recalculation.
- The hosted `gpt-4o` snapshot and API seed were not pinned during the reported
  runs. Exact model-output reproduction is therefore not guaranteed; the
  preserved per-case outputs are the record of the reported evaluation.
- The large BIOS knowledge-graph and vector-index files are not distributed in
  this repository. The preserved C3 scored file supports rerunning Phase 2 and
  rescoring without rebuilding Phase 1.

## Canonical artifacts

- `data/benchmark_500.csv`
- `outputs/ablation_500/C3_benchmark500_scored.csv`
- `outputs/ablation_500/C4_benchmark500_v31_scored.csv`
- `outputs/ablation_500/C4_benchmark500_v31_phase2.jsonl`
- `outputs/ablation_500/ablation_report_benchmark500_FINAL.md`

The reported benchmark_500 C4 result is F1 94.3%, precision 89.3%, recall
100.0%, and FPR 11.1%, with TP=158, FN=0, FP=19, and TN=152. These values are
copied from the frozen final report and are not recalculated here.

## Environment and execution

The implementation is under `nyuolab-src-clean/` and requires Python 3.11.

```bash
cd nyuolab-src-clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An OpenAI API key is required for a new Phase 2 run. See
`tools/run_ablation_500.sh` for the recorded pipeline entry point. The script's
full Phase 1 route additionally requires the BIOS KG/vector resources described
above.

Build the local code and dataset deposit candidates with:

```bash
python tools/build_jmir_v31_release.py
```

The generated ZIP files and their SHA-256 hashes are written to `dist/`.

## Licensing and deposit status

The implementation in `nyuolab-src-clean/` is distributed under GPL-3.0; see
`nyuolab-src-clean/LICENSE`. The benchmark dataset is distributed under
CC BY-NC 4.0; see `data/LICENSE_CC_BY_NC_4.0.md`. The DOI and final
public-repository SHA are not assigned until the repository and deposits are
approved and archived.

## Provenance

The implementation originated from the codebase associated with “Medical Large
Language Models are Vulnerable to Data Poisoning Attacks” and was modified for
the two-stage pipeline evaluated in the manuscript above. The original
attribution is retained in `nyuolab-src-clean/README.md`.
