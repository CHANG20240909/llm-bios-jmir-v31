# Two-Stage KG–LLM Medical-Fact Safety Screening

This directory contains the implementation used by the LLM-BIOS JMIR AI v3.1
reproducibility snapshot. It extends the code associated with
["Medical Large Language Models are Vulnerable to Data Poisoning
Attacks"](https://www.nature.com/articles/s41591-024-03445-1)
(*Nature Medicine*, 2024).

The evaluated C4 configuration consists of:

1. Phase 1 C3 knowledge-graph screening with semantic and modifier-aware
   handling;
2. deterministic Phase 1.5 modifier reconciliation; and
3. Phase 2 GPT-4o verification with the v3.1 prompt, G1–G4 guardrails, and
   deterministic post-processing.

The `SYSTEM_PROMPT` in `llm_agent.py` is the evaluated v3.1 prompt. Later v3.2
or evidence-augmentation experiments are intentionally excluded from this
release branch.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

New Phase 2 calls require `OPENAI_API_KEY`. Do not commit API keys or `.env`
files.

## Reproduction boundary

The large BIOS KG and vector index are not included. The repository preserves
the benchmark, C3 Phase 1 predictions, v3.1 Phase 2 outputs, final scored file,
and final report. These artifacts allow the reported outputs to be audited and
permit Phase 2 reruns from the preserved C3 alarms, subject to hosted-model
variability.

Embedding models and external biomedical resources retain their own licenses.
The code in this directory is licensed under GPL-3.0; see `LICENSE`.
