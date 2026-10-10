# JMIR AI v3.1 Release Manifest

Status: local pre-release candidate; not yet published or assigned a DOI.

## Version identity

- Candidate release: `v3.1.1`
- Public repository: `CHANG20240909/llm-bios-jmir-v31` (planned)
- Source release branch: `release/jmir-ai-v1.0.0-v31`
- Development-repository base: `b7a786d6`
- Evaluated prompt source: `ec0791be6305f1062aabf0810a55caed4e08db5e`
- Evaluated configuration: C4 v3.1
- Inference date reported in the manuscript: April 18, 2026

This sanitized snapshot restores `nyuolab-src-clean/llm_agent.py` from the
evaluated v3.1 prompt source. The v3.2 G4 change first introduced at
`02e286ab5d76b9a51cd9bfb7ff0eaf09b932f370` is excluded.

## Code-deposit candidate

The code ZIP built by `tools/build_jmir_v31_release.py` contains:

- the release README and this manifest;
- the complete tracked `nyuolab-src-clean/` implementation and GPL-3.0 license;
- ablation configuration files;
- the benchmark phrase input;
- the preserved C3 Phase 1 scored output;
- the frozen C4 v3.1 per-case output, scored output, and final report; and
- the recorded ablation and metric scripts.

The large BIOS KG and vector index are not included. Exact reruns of hosted
GPT-4o output are not guaranteed because the API seed and model snapshot were
not pinned.

## Dataset-deposit candidate

The dataset ZIP contains:

- `data/benchmark_500.csv`; and
- `data/README_benchmark_500.md`; and
- `data/LICENSE_CC_BY_NC_4.0.md`.

The dataset license is CC BY-NC 4.0. The DOI and final citation remain pending.

## Metadata correction

Release `v3.1.1` supersedes `v3.1.0` as a metadata-only correction. The
`v3.1.0` GitHub tag and Zenodo record correctly identified the release version,
but the archived `CITATION.cff` and this manifest still carried the earlier
candidate version `1.0.0`. Release `v3.1.1` aligns those metadata fields. The
evaluated v3.1 prompt, code behavior, benchmark, canonical outputs, and reported
results are unchanged.

## Files protected from modification

Release packaging reads but does not modify:

- `data/benchmark_500.csv`
- `outputs/ablation_500/C4_benchmark500_v31_scored.csv`
- `outputs/ablation_500/C4_benchmark500_v31_phase2.jsonl`
- `outputs/ablation_500/ablation_report_benchmark500_FINAL.md`

## Publication gates

Before publication:

1. complete author degrees, CRediT roles, conflict-of-interest declarations,
   and final approval;
2. review the branch for secrets and unintended files;
3. authenticate GitHub and publish an approved public release repository;
4. reserve and insert the code and dataset DOIs;
5. record the final full release commit SHA; and
6. generate the official TRIPOD-LLM PDF as Multimedia Appendix 7.
