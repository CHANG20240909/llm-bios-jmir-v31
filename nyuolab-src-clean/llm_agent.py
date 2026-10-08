"""
Phase 2 LLM Agent — GPT-4o-based auxiliary verification for alarm cases.

Receives WARN/FAIL cases from Phase 1, constructs a structured prompt with
KG candidate information, and returns a structured JSON verdict.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI


@dataclass
class AlarmCase:
    """An alarm case from Phase 1 for LLM review."""
    source_sentence: str = ""
    origin: str = ""
    relationship: str = ""
    target: str = ""
    phase1_alarm: str = ""
    phase1_fail_locations: list = field(default_factory=list)
    origin_status: str = ""
    origin_candidates: list = field(default_factory=list)
    target_status: str = ""
    target_candidates: list = field(default_factory=list)


@dataclass
class LLMVerdict:
    """Structured LLM verdict."""
    verdict: str = "WARN"
    reasoning: str = ""
    origin_assessment: str = "n/a"
    target_assessment: str = "n/a"
    modifier_changes_meaning: bool = False
    direction_correct: bool = True
    evidence_tier: str = "WEAK"
    suggested_origin: str = ""
    suggested_target: str = ""
    error: str = ""


SYSTEM_PROMPT = """You are a medical knowledge verification assistant operating
inside a SAFETY screening pipeline. You only review triples that Phase 1 already
flagged as WARN or FAIL — you NEVER see triples Phase 1 considered safe (PASS).

Your role is asymmetric: clearing an alarm on a HARMFUL triple (false negative)
is strictly worse than keeping a false alarm (false positive). When in doubt,
KEEP the alarm.

============================================================
PASS GUARDRAILS — ALL must be satisfied to issue verdict=PASS
============================================================

[G1] DIRECTION CHECK (most common failure pattern)
The triple "A -- R -- B" means A is the SUBJECT performing R on B.
Before clearing the alarm, ask:
  - Is the medically correct fact "A does R to B" or "B does R to A"?
  - If the direction is reversed, the triple is INCORRECT regardless of
    whether the underlying relationship exists.

Common reversal patterns:
  - may diagnose: diagnostic test/sign --> disease  (NOT disease --> test)
      e.g. "Tuberculosis may diagnose sputum culture" is REVERSED
  - may treat:    drug/intervention --> disease     (NOT disease --> drug)
      e.g. "Eczema may treat topical corticosteroids" is REVERSED
      e.g. "Lupus nephritis may treat Cyclophosphamide" is REVERSED — a disease
      cannot be the subject of "may treat" with a drug as the object; read the
      triple literally (Origin -- relationship --> Target).
  - may cause:    cause --> effect                   (direction-sensitive)
  - has adverse effect: drug --> side effect        (NOT side effect --> drug)
  - contraindication: drug --> condition (a condition in which the drug should not be given)
  - interacts with:   drug --> drug (NOT disease --> drug or drug --> disease)

SYMMETRIC RELATIONS (direction is NOT a failure condition):
  - ddx (differential diagnosis): clinically symmetric; "A -- ddx -- B" and
    "B -- ddx -- A" are both acceptable if A and B are legitimate differentials
    for the same presenting complaint. Do NOT mark direction_correct=false on
    ddx triples unless one side is not a disease/condition at all.

Set direction_correct=false and verdict=WARN/FAIL only for truly asymmetric
relations that are reversed.

[G2] CO-OCCURRENCE IS NOT INTERACTION
"interacts with" requires a documented pharmacokinetic or pharmacodynamic
mechanism (CYP induction/inhibition, receptor competition, additive toxicity,
QT prolongation, etc.).
Two drugs being used together for the same condition does NOT establish a
drug interaction. Co-prescription != interaction.

[G3] NO GENERALIZATION FROM PRINCIPLES TO SPECIFIC CLAIMS
Do NOT clear a "contraindication" alarm by invoking a general principle
(e.g. "drug X is hepatically metabolized, so liver disease is a
contraindication"). The specific labeling matters:
  - Authoritative sources (FDA label, UpToDate, NCCN) must list the exact
    relationship as a CONTRAINDICATION (not "use with caution",
    "dose adjustment", "monitor", or "precaution").
  - "Caution" / "precaution" / "use with care" / "consider therapy
    modification" is NOT the same as "contraindication".
  - If the actual clinical effect is OPPOSITE to what the triple claims
    (e.g., hemodialysis CORRECTS hypokalemia, not contraindicated by it),
    KEEP the alarm.

[G4] EVIDENCE TIER REQUIREMENT
You may only set verdict=PASS when evidence_tier in {GUIDELINE, LABEL, TEXTBOOK}.
If your reasoning is MECHANISTIC inference or WEAK association only, the
verdict must be WARN at most.
Tiers (strongest to weakest):
  - GUIDELINE: NCCN/WHO/major society guideline, FDA-approved indication
  - LABEL:     FDA label, UpToDate explicit recommendation
  - TEXTBOOK:  Standard, widely accepted medical fact taught in core training
               (e.g. NSAIDs can worsen CKD; HIV and mononucleosis share
               symptoms on ddx; statins can cause myopathy). Sufficient for PASS.
  - MECHANISTIC: Plausible by mechanism but not directly documented
  - WEAK:      Indirect or speculative

Pick the HIGHEST tier that applies. If a relationship is explicitly listed in
an FDA label or UpToDate recommendation, use LABEL (not TEXTBOOK). If it is
covered by a guideline, use GUIDELINE. Do not downgrade well-established
facts to MECHANISTIC just because you cannot cite a specific document.

============================================================
RESPONSE FORMAT
============================================================

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{
  "verdict": "PASS" | "WARN" | "FAIL",
  "reasoning": "<2-4 sentences. State your direction check (G1) and evidence basis (G4) explicitly.>",
  "direction_correct": true | false,
  "evidence_tier": "GUIDELINE" | "LABEL" | "TEXTBOOK" | "MECHANISTIC" | "WEAK",
  "origin_assessment": "synonym" | "superclass" | "subclass" | "unrelated" | "n/a",
  "target_assessment": "synonym" | "superclass" | "subclass" | "unrelated" | "n/a",
  "modifier_changes_meaning": true | false,
  "suggested_origin": "<if you can suggest a better KG concept, else empty>",
  "suggested_target": "<if you can suggest a better KG concept, else empty>"
}

DECISION RULES (applied AFTER you fill the JSON above):
- If direction_correct == false  -> verdict must be WARN or FAIL (never PASS)
- If evidence_tier in {MECHANISTIC, WEAK} -> verdict must be WARN at most (never PASS)
- verdict=FAIL: triple contradicts medical knowledge or violates G1/G2/G3
- verdict=WARN: triple is plausible but does not meet PASS bar
- verdict=PASS: G1/G2/G3 all satisfied AND evidence_tier in {GUIDELINE, LABEL, TEXTBOOK}
"""


def build_user_prompt(case: AlarmCase) -> str:
    """Construct the user prompt from an alarm case."""
    parts = [
        f"## Medical Triple Under Review",
        f"- Origin: {case.origin}",
        f"- Relationship: {case.relationship}",
        f"- Target: {case.target}",
        f"",
        f"## Phase 1 Screening Result",
        f"- Alarm level: {case.phase1_alarm}",
        f"- Fail locations: {', '.join(case.phase1_fail_locations) if case.phase1_fail_locations else 'path (no component-level failure)'}",
    ]

    if case.origin_status in ("WARN", "FAIL"):
        parts.append(f"")
        parts.append(f"## Origin Analysis")
        parts.append(f"- Input: \"{case.origin}\"")
        parts.append(f"- Status: {case.origin_status}")
        parts.append(f"- Top KG candidates: {case.origin_candidates}")

    if case.target_status in ("WARN", "FAIL"):
        parts.append(f"")
        parts.append(f"## Target Analysis")
        parts.append(f"- Input: \"{case.target}\"")
        parts.append(f"- Status: {case.target_status}")
        parts.append(f"- Top KG candidates: {case.target_candidates}")

    if case.source_sentence:
        parts.append(f"")
        parts.append(f"## Original Clinical Text")
        parts.append(f"\"{case.source_sentence}\"")

    return "\n".join(parts)


def call_llm(case: AlarmCase, model: str = "gpt-4o", max_retries: int = 3) -> LLMVerdict:
    """Call GPT-4o and parse the structured JSON response."""
    client = OpenAI()
    user_prompt = build_user_prompt(case)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)

            verdict = data.get("verdict", "WARN")
            direction_correct = bool(data.get("direction_correct", True))
            evidence_tier = str(data.get("evidence_tier", "WEAK")).upper()

            if verdict == "PASS":
                if not direction_correct:
                    verdict = "WARN"
                elif evidence_tier not in ("GUIDELINE", "LABEL", "TEXTBOOK"):
                    verdict = "WARN"

            return LLMVerdict(
                verdict=verdict,
                reasoning=data.get("reasoning", ""),
                origin_assessment=data.get("origin_assessment", "n/a"),
                target_assessment=data.get("target_assessment", "n/a"),
                modifier_changes_meaning=data.get("modifier_changes_meaning", False),
                direction_correct=direction_correct,
                evidence_tier=evidence_tier,
                suggested_origin=data.get("suggested_origin", ""),
                suggested_target=data.get("suggested_target", ""),
            )

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return LLMVerdict(error=f"JSON parse error after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt == max_retries - 1:
                return LLMVerdict(error=f"API error after {max_retries} attempts: {e}")

    return LLMVerdict(error="max retries exhausted")


def dry_run_verdict(case: AlarmCase) -> LLMVerdict:
    """Return a mock verdict without calling the API (for testing)."""
    return LLMVerdict(
        verdict=case.phase1_alarm,
        reasoning=f"[DRY RUN] echoing Phase 1 alarm level: {case.phase1_alarm}",
        origin_assessment="n/a",
        target_assessment="n/a",
    )
