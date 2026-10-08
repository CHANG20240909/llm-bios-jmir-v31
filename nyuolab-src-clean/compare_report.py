"""
4-Way Comparison Report: Baseline vs Phase 1 vs Phase 1.5 (modifier) vs Phase 2 (LLM)

Reads Phase 1 scored CSVs (with optional phase15_alarm_level column) and
Phase 2 JSONL results, then produces a structured Markdown comparison report
showing how each layer reduces expert review burden.

Usage:
    python compare_report.py --output ../outputs/comparison_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to nyuolab-src-clean/)
# ---------------------------------------------------------------------------
PHASE1_LLM_ANSWER_CSV   = "../data/phase2a5_llm_answer_phrases_scored.csv"
PHASE1_LLM_EASY_CSV     = "../data/llm_answer_easy_phrases_scored.csv"
PHASE2_JSONL            = "../outputs/phase2_llm_agent_results.jsonl"


def load_phase2_results(jsonl_path):
    """Load Phase 2 JSONL into a dict keyed by (origin, relationship, target)."""
    results = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                key = (rec["origin"], rec["relationship"], rec["target"])
                results[key] = rec
    except FileNotFoundError:
        pass
    return results


def baseline_verdict(row):
    """Baseline: pure path_exists — True = Non-harmful, False = Harmful."""
    return "PASS" if str(row.get("result", "False")).lower() in ("true", "1") else "FAIL"


def phase1_verdict(row):
    return str(row.get("alarm_level", "FAIL"))


def phase15_verdict(row):
    """Phase 1.5: use phase15_alarm_level if present, else same as Phase 1."""
    v = row.get("phase15_alarm_level", "")
    return str(v) if v and str(v) not in ("", "nan") else phase1_verdict(row)


def phase15_modifier_info(row):
    """Return a short modifier summary string, or empty string."""
    if str(row.get("modifier_stripped", "False")).lower() != "true":
        return ""
    core = row.get("modifier_core", "")
    cat = row.get("modifier_category", "")
    upgraded = str(row.get("modifier_alarm_upgraded", "False")).lower() == "true"
    path = str(row.get("modifier_core_path_found", "False")).lower() == "true"
    if upgraded:
        return f"↑UPGRADED core={core} [{cat}]"
    elif path:
        return f"PATH core={core} [{cat}]"
    else:
        return f"core={core} [{cat}]"


def analyze_dataset(csv_path, phase2_results, dataset_name):
    """Analyze one dataset across all four layers."""
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        key = (str(row["origin"]), str(row["relationship"]), str(row["target"]))
        p2 = phase2_results.get(key)

        base  = baseline_verdict(row)
        p1    = phase1_verdict(row)
        p15   = phase15_verdict(row)
        p15_info = phase15_modifier_info(row)
        p2v   = p2["llm_verdict"] if p2 else "N/A"
        p2r   = p2["llm_reasoning"] if p2 else ""
        fail_locs = str(row.get("fail_locations", "")) if pd.notna(row.get("fail_locations")) else ""

        rows.append({
            "origin":       row["origin"],
            "relationship": row["relationship"],
            "target":       row["target"],
            "baseline":     base,
            "phase1":       p1,
            "phase15":      p15,
            "phase15_info": p15_info,
            "phase2":       p2v,
            "p1_fail_locs": fail_locs,
            "p1_origin_matched":  str(row.get("origin_matched", "")),
            "p1_origin_dist":     row.get("origin_distance", ""),
            "p1_target_matched":  str(row.get("target_matched", "")),
            "p1_target_dist":     row.get("target_distance", ""),
            "p2_reasoning": p2r,
            "p2_origin_assess": p2["llm_origin_assessment"] if p2 else "",
            "p2_target_assess": p2["llm_target_assessment"] if p2 else "",
            "p2_modifier": p2["llm_modifier_changes_meaning"] if p2 else "",
        })

    return rows


def count_verdicts(rows, col):
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "N/A": 0}
    for r in rows:
        v = r[col]
        counts[v] = counts.get(v, 0) + 1
    return counts


def burden_reduction(rows):
    """Calculate expert review burden at each stage."""
    total = len(rows)
    base_review   = sum(1 for r in rows if r["baseline"] == "FAIL")
    p1_review     = sum(1 for r in rows if r["phase1"] in ("WARN", "FAIL"))
    p15_review    = sum(1 for r in rows if r["phase15"] in ("WARN", "FAIL"))
    p15_upgraded  = sum(1 for r in rows if r["phase15"] != r["phase1"])
    p2_review     = sum(1 for r in rows if r["phase2"] in ("WARN", "FAIL", "N/A"))
    p2_available  = sum(1 for r in rows if r["phase2"] != "N/A")
    p2_confirmed_pass = sum(1 for r in rows if r["phase2"] == "PASS")

    return {
        "total": total,
        "base_review": base_review,
        "p1_review": p1_review,
        "p15_review": p15_review,
        "p15_upgraded": p15_upgraded,
        "p2_available": p2_available,
        "p2_confirmed_pass": p2_confirmed_pass,
        "p2_review": p2_review,
    }


def format_triple(r):
    return f"`{r['origin']}` → `{r['relationship']}` → `{r['target']}`"


def verdict_icon(v):
    return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "N/A": "➖"}.get(v, "❓")


def build_report(rows_llm, rows_easy):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "# 4-Way Comparison Report",
        f"> Generated: {now}",
        "",
        "## Summary",
        "",
        "| System | 판단 방식 | 정보 제공 | 전문가 부담 |",
        "|--------|----------|----------|------------|",
        "| **Baseline** | `path_exists` True/False | 결과만 | 전수 검토 |",
        "| **Phase 1** | PASS / WARN / FAIL (3단계) | 실패 위치 + KG 후보 | alarm 케이스만 |",
        "| **Phase 1.5** | 수식어 분리 + core 재검색 | modifier 정보 + core 거리 | staging 개선 건 제외 |",
        "| **Phase 2** | GPT-4o 보조 재판정 | 의학적 근거 + 제안 | WARN/FAIL 중 불확실 건만 |",
        "",
    ]

    for dataset_name, rows in [("llm_answer (최신 의료 트리플, 6개)", rows_llm),
                                 ("llm_answer_easy (KG-friendly 트리플, 9개)", rows_easy)]:
        b = count_verdicts(rows, "baseline")
        p1 = count_verdicts(rows, "phase1")
        p2 = count_verdicts(rows, "phase2")
        brd = burden_reduction(rows)

        lines += [
            f"---",
            f"",
            f"## 데이터셋: {dataset_name}",
            f"",
            f"### 판정 요약",
            f"",
            f"| 트리플 | Baseline | Phase 1 | Phase 1.5 | Phase 2 (GPT-4o) |",
            f"|--------|----------|---------|-----------|-----------------|",
        ]

        for r in rows:
            triple   = f"{r['origin']} → {r['relationship']} → {r['target']}"
            b_icon   = verdict_icon(r["baseline"])
            p1_icon  = verdict_icon(r["phase1"])
            p15_icon = verdict_icon(r["phase15"])
            p2_icon  = verdict_icon(r["phase2"])
            p15_note = f" _{r['phase15_info']}_" if r.get("phase15_info") else ""
            lines.append(
                f"| {triple} "
                f"| {b_icon} {r['baseline']} "
                f"| {p1_icon} {r['phase1']} "
                f"| {p15_icon} {r['phase15']}{p15_note} "
                f"| {p2_icon} {r['phase2']} |"
            )

        lines += [
            f"",
            f"### 전문가 검토 부담 비교",
            f"",
            f"| 단계 | 검토 필요 건수 / 전체 | 비율 |",
            f"|------|----------------------|------|",
            f"| Baseline | {brd['base_review']} / {brd['total']} | {brd['base_review']/brd['total']*100:.0f}% |",
            f"| Phase 1  | {brd['p1_review']} / {brd['total']} | {brd['p1_review']/brd['total']*100:.0f}% |",
            f"| Phase 1.5| {brd['p15_review']} / {brd['total']} | {brd['p15_review']/brd['total']*100:.0f}% |",
        ]
        if brd['p15_upgraded'] > 0:
            lines.append(
                f"\n> Phase 1.5: alarm **{brd['p15_upgraded']}건 FAIL→WARN 개선** (staging 수식어 분리)"
            )
        if brd['p2_available'] > 0:
            lines.append(
                f"| Phase 2  | {brd['p2_review']} / {brd['p2_available']} (alarm 중) "
                f"| {brd['p2_review']/brd['p2_available']*100:.0f}% |"
            )
            if brd['p2_confirmed_pass'] > 0:
                lines.append(
                    f"\n> GPT-4o가 Phase 1.5 alarm {brd['p2_available']}건 중 "
                    f"**{brd['p2_confirmed_pass']}건을 PASS로 재판정** → 전문가 검토 대상 제외 가능"
                )
        else:
            lines.append(f"| Phase 2  | N/A (alarm 없음) | 0% |")

        lines.append("")

    # 전체 종합 분석
    all_rows = rows_llm + rows_easy
    total = len(all_rows)
    base_total   = sum(1 for r in all_rows if r["baseline"] == "FAIL")
    p1_total     = sum(1 for r in all_rows if r["phase1"] in ("WARN", "FAIL"))
    p15_total    = sum(1 for r in all_rows if r["phase15"] in ("WARN", "FAIL"))
    p15_upgraded = sum(1 for r in all_rows if r["phase15"] != r["phase1"])
    p2_alarm     = sum(1 for r in all_rows if r["phase2"] != "N/A")
    p2_pass      = sum(1 for r in all_rows if r["phase2"] == "PASS")
    p2_review    = sum(1 for r in all_rows if r["phase2"] in ("WARN", "FAIL"))

    lines += [
        "---",
        "",
        "## 전체 종합 분석",
        "",
        f"- 전체 트리플: **{total}개**",
        f"- Baseline 전문가 검토 필요: **{base_total}/{total}** ({base_total/total*100:.0f}%)",
        f"- Phase 1 이후 검토 필요: **{p1_total}/{total}** ({p1_total/total*100:.0f}%)",
        f"- Phase 1.5 이후 검토 필요: **{p15_total}/{total}** ({p15_total/total*100:.0f}%)"
        + (f"  _(alarm {p15_upgraded}건 개선)_" if p15_upgraded > 0 else "  _(수식어 분리 미적용 또는 개선 없음)_"),
    ]
    if p2_alarm > 0:
        lines += [
            f"- Phase 2 처리 대상 (alarm): **{p2_alarm}건**",
            f"  - GPT-4o PASS 재판정: **{p2_pass}건** → 전문가 검토 불필요",
            f"  - 최종 전문가 검토 필요: **{p2_review}건**",
            f"",
            f"### 핵심 발견",
            f"",
        ]
        if p2_pass > 0:
            pass_cases = [r for r in all_rows if r["phase2"] == "PASS"]
            for r in pass_cases:
                lines.append(
                    f"- **{r['origin']}** → {r['relationship']} → {r['target']}: "
                    f"Phase 1 {r['phase1']} → GPT-4o **PASS** 재판정"
                )
                if r["p2_reasoning"]:
                    lines.append(f"  > {r['p2_reasoning']}")

        modifier_cases = [r for r in all_rows if str(r.get("p2_modifier", "")).lower() == "true"]
        if modifier_cases:
            lines.append("")
            lines.append("- **수식어가 의미를 바꾸는 케이스** (modifier_changes_meaning=True):")
            for r in modifier_cases:
                lines.append(f"  - `{r['target']}` — GPT-4o 판단: modifier가 임상적 의미 변경")

    lines += [
        "",
        "---",
        "",
        "## Phase 1 상세: fail_locations 분석",
        "",
        "| 트리플 | fail_locations | origin 매칭 | target 매칭 |",
        "|--------|---------------|------------|------------|",
    ]
    for r in rows_llm:
        if r["phase1"] in ("WARN", "FAIL"):
            o_info = f"{r['p1_origin_matched']} (dist={float(r['p1_origin_dist']):.3f})" if r["p1_origin_dist"] != "" else "-"
            t_info = f"{r['p1_target_matched']} (dist={float(r['p1_target_dist']):.3f})" if r["p1_target_dist"] != "" else "-"
            lines.append(
                f"| {r['origin']} → {r['target']} "
                f"| `{r['p1_fail_locs']}` "
                f"| {o_info} "
                f"| {t_info} |"
            )

    # Phase 1.5 상세
    p15_cases = [r for r in rows_llm if r.get("phase15_info")]
    if p15_cases:
        lines += [
            "",
            "---",
            "",
            "## Phase 1.5 상세: 수식어 분리 결과",
            "",
            "| 트리플 | 수식어 분리 정보 | alarm 변경 |",
            "|--------|-----------------|-----------|",
        ]
        for r in p15_cases:
            upgraded = "FAIL→WARN" if "UPGRADED" in r.get("phase15_info", "") else "-"
            lines.append(
                f"| {r['origin']} → {r['target']} "
                f"| {r.get('phase15_info', '')} "
                f"| {upgraded} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Phase 2 상세: LLM 판단 근거",
        "",
    ]
    for r in rows_llm:
        if r["phase2"] not in ("N/A",):
            p1_icon  = verdict_icon(r["phase1"])
            p15_icon = verdict_icon(r["phase15"])
            p2_icon  = verdict_icon(r["phase2"])
            p15_line = f"- Phase 1.5: {p15_icon} `{r['phase15']}`" + (f"  _{r['phase15_info']}_" if r.get("phase15_info") else "")
            lines += [
                f"### {r['origin']} → {r['relationship']} → {r['target']}",
                f"- Phase 1: {p1_icon} `{r['phase1']}`  →  Phase 2: {p2_icon} `{r['phase2']}`",
                p15_line,
                f"- 근거: {r['p2_reasoning']}",
                f"- origin 평가: `{r['p2_origin_assess']}`  /  target 평가: `{r['p2_target_assess']}`",
                f"- modifier 의미 변경: `{r['p2_modifier']}`",
                "",
            ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate 3-way comparison report")
    parser.add_argument("--output", default="../outputs/comparison_report.md",
                        help="Output markdown path (default: ../outputs/comparison_report.md)")
    args = parser.parse_args()

    phase2_results = load_phase2_results(PHASE2_JSONL)

    rows_llm  = analyze_dataset(PHASE1_LLM_ANSWER_CSV,  phase2_results, "llm_answer")
    rows_easy = analyze_dataset(PHASE1_LLM_EASY_CSV,    phase2_results, "llm_answer_easy")

    report = build_report(rows_llm, rows_easy)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] Report saved: {out_path}")
    print(f"     llm_answer   : {len(rows_llm)} rows")
    print(f"     llm_easy     : {len(rows_easy)} rows")
    print(f"     Phase 2 hits : {len(phase2_results)} records in JSONL")


if __name__ == "__main__":
    main()
