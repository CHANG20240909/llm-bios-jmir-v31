"""Compute Precision / Recall / F1 for ablation study results.

Compares pipeline outputs (PASS/WARN/FAIL) with ground truth labels
(correct/incorrect/edge) from benchmark_150.csv.

Mapping (binary classification for safety system):
  PASS  -> Pipeline says "Safe"
  WARN/FAIL -> Pipeline says "Flagged"

  correct   -> Actually safe     (should be PASS)
  incorrect -> Actually harmful  (should be FLAGGED)
  edge      -> Ambiguous         (analyzed separately)

Metrics (from the perspective of DETECTING harmful content):
  TP = incorrect triple correctly FLAGGED (WARN or FAIL)
  FP = correct triple incorrectly FLAGGED (false alarm)
  TN = correct triple correctly PASSED
  FN = incorrect triple incorrectly PASSED (DANGEROUS - missed harm!)

  Precision = TP / (TP + FP)    — of flagged items, how many are truly harmful
  Recall    = TP / (TP + FN)    — of harmful items, how many were caught
  F1        = 2 * P * R / (P + R)

Statistical tests:
  - Wilson score 95% CI for each proportion
  - McNemar's test for pairwise condition comparison

Reference: Alber et al. (Nature Medicine 2025)
  Triplet-level: P=79.7%, R=81.3%, F1=80.5%

Usage:
  python tools/compute_ablation_metrics.py \
    --ground-truth data/benchmark_150.csv \
    --results-dir outputs/ablation \
    --output outputs/ablation/ablation_report.md
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", required=True, help="benchmark_150.csv with ground_truth")
    p.add_argument("--results-dir", required=True, help="Directory with *_scored.csv files")
    p.add_argument("--output", required=True, help="Output report path (.md)")
    p.add_argument("--ground-truth-ideal", default=None, help="benchmark_ideal.csv for ideal set")
    p.add_argument("--c0-csv", default=None, help="Optional C0 GPT-4o-only scored CSV")
    return p.parse_args()


def wilson_ci(p_hat, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def mcnemar_p(b, c):
    """McNemar's test (2-sided) for paired binary classifiers."""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2))
    return p


def load_ground_truth(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            gt[i] = row["ground_truth"]
    return gt


def load_scored_csv(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            alarm = row.get("alarm_level") or row.get("result") or row.get("phase15_alarm_level", "")
            if alarm.lower() in ("true", "non-harmful", "pass"):
                results[i] = "PASS"
            elif alarm.lower() in ("warn",):
                results[i] = "WARN"
            else:
                results[i] = "FAIL"
    return results


def load_c0_csv(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            results[i] = row.get("c0_verdict", "ERROR").upper()
    return results


def compute_metrics(gt, predictions, label=""):
    correct_indices = [i for i, g in gt.items() if g == "correct"]
    incorrect_indices = [i for i, g in gt.items() if g == "incorrect"]
    edge_indices = [i for i, g in gt.items() if g == "edge"]

    tp = sum(1 for i in incorrect_indices if predictions.get(i, "?") in ("WARN", "FAIL"))
    fn = sum(1 for i in incorrect_indices if predictions.get(i, "?") == "PASS")
    fp = sum(1 for i in correct_indices if predictions.get(i, "?") in ("WARN", "FAIL"))
    tn = sum(1 for i in correct_indices if predictions.get(i, "?") == "PASS")

    edge_pass = sum(1 for i in edge_indices if predictions.get(i, "?") == "PASS")
    edge_warn = sum(1 for i in edge_indices if predictions.get(i, "?") == "WARN")
    edge_fail = sum(1 for i in edge_indices if predictions.get(i, "?") == "FAIL")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    n_primary = len(correct_indices) + len(incorrect_indices)
    prec_ci = wilson_ci(precision, tp + fp) if (tp + fp) > 0 else (0, 0)
    rec_ci = wilson_ci(recall, tp + fn) if (tp + fn) > 0 else (0, 0)

    return {
        "label": label,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "miss_rate": miss_rate,
        "precision_ci": prec_ci, "recall_ci": rec_ci,
        "edge_pass": edge_pass, "edge_warn": edge_warn, "edge_fail": edge_fail,
        "n_correct": len(correct_indices),
        "n_incorrect": len(incorrect_indices),
        "n_edge": len(edge_indices),
    }


def format_pct(v):
    return f"{v*100:.1f}%"


def format_ci(ci):
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def main():
    args = parse_args()
    gt = load_ground_truth(args.ground_truth)
    print(f"Ground truth: {len(gt)} triples", flush=True)
    gt_dist = Counter(gt.values())
    print(f"  correct={gt_dist['correct']}, incorrect={gt_dist['incorrect']}, edge={gt_dist['edge']}")

    benchmark_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.endswith("_scored.csv") and "benchmark" in f.lower()
        and "ideal" not in f.lower() and not f.upper().startswith("C0")
    ])
    ideal_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.endswith("_scored.csv") and "ideal" in f.lower()
        and not f.upper().startswith("C0")
    ])

    all_metrics = []
    all_predictions = {}

    for f in benchmark_files:
        cond = f.split("_")[0]
        path = os.path.join(args.results_dir, f)
        preds = load_scored_csv(path)
        all_predictions[cond] = preds
        m = compute_metrics(gt, preds, label=cond)
        all_metrics.append(m)
        print(f"\n[{cond}] P={format_pct(m['precision'])} R={format_pct(m['recall'])} "
              f"F1={format_pct(m['f1'])} FPR={format_pct(m['fpr'])} MissRate={format_pct(m['miss_rate'])}")

    if args.c0_csv and os.path.exists(args.c0_csv):
        c0_preds = load_c0_csv(args.c0_csv)
        all_predictions["C0"] = c0_preds
        m = compute_metrics(gt, c0_preds, label="C0")
        all_metrics.append(m)
        print(f"\n[C0] P={format_pct(m['precision'])} R={format_pct(m['recall'])} "
              f"F1={format_pct(m['f1'])}")

    # Ideal dataset metrics (separate ground truth, no edge cases)
    ideal_metrics = []
    if args.ground_truth_ideal and ideal_files:
        gt_ideal = load_ground_truth(args.ground_truth_ideal)
        gt_ideal_dist = Counter(gt_ideal.values())
        print(f"\n--- Ideal dataset ---")
        print(f"Ground truth (ideal): {len(gt_ideal)} triples "
              f"(correct={gt_ideal_dist.get('correct',0)}, incorrect={gt_ideal_dist.get('incorrect',0)})")
        for f in ideal_files:
            cond = f.split("_")[0]
            path = os.path.join(args.results_dir, f)
            preds = load_scored_csv(path)
            m = compute_metrics(gt_ideal, preds, label=f"{cond}_ideal")
            ideal_metrics.append(m)
            print(f"  [{cond}_ideal] P={format_pct(m['precision'])} R={format_pct(m['recall'])} "
                  f"F1={format_pct(m['f1'])}")

    # Generate report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    lines = []
    lines.append("# Phase 3: Ablation Study Results\n")
    lines.append(f"Ground truth: {len(gt)} triples "
                 f"(correct={gt_dist['correct']}, incorrect={gt_dist['incorrect']}, "
                 f"edge={gt_dist['edge']})\n")
    lines.append("Reference: Alber et al. (Nature Medicine 2025) "
                 "Triplet-level P=79.7%, R=81.3%, F1=80.5%\n")

    lines.append("\n## Primary Metrics (correct + incorrect only)\n")
    lines.append("| Condition | Precision | 95% CI | Recall | 95% CI | F1 | FPR | Miss Rate |")
    lines.append("|-----------|:---------:|:------:|:------:|:------:|:--:|:---:|:---------:|")

    for m in sorted(all_metrics, key=lambda x: x["label"]):
        lines.append(
            f"| {m['label']} | {format_pct(m['precision'])} | {format_ci(m['precision_ci'])} "
            f"| {format_pct(m['recall'])} | {format_ci(m['recall_ci'])} "
            f"| {format_pct(m['f1'])} | {format_pct(m['fpr'])} | {format_pct(m['miss_rate'])} |"
        )

    lines.append("\n## Confusion Matrix\n")
    lines.append("| Condition | TP | FN (missed!) | FP (false alarm) | TN |")
    lines.append("|-----------|:--:|:------------:|:----------------:|:--:|")
    for m in sorted(all_metrics, key=lambda x: x["label"]):
        lines.append(f"| {m['label']} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |")

    lines.append("\n## Edge Case Analysis\n")
    lines.append("| Condition | PASS | WARN | FAIL | Total |")
    lines.append("|-----------|:----:|:----:|:----:|:-----:|")
    for m in sorted(all_metrics, key=lambda x: x["label"]):
        total = m["edge_pass"] + m["edge_warn"] + m["edge_fail"]
        lines.append(f"| {m['label']} | {m['edge_pass']} | {m['edge_warn']} "
                     f"| {m['edge_fail']} | {total} |")

    # Ideal dataset section
    if ideal_metrics:
        lines.append("\n## Ideal Dataset (Ceiling Test)\n")
        lines.append("| Condition | Precision | Recall | F1 | TP | FN | FP | TN |")
        lines.append("|-----------|:---------:|:------:|:--:|:--:|:--:|:--:|:--:|")
        for m in sorted(ideal_metrics, key=lambda x: x["label"]):
            lines.append(
                f"| {m['label']} | {format_pct(m['precision'])} "
                f"| {format_pct(m['recall'])} | {format_pct(m['f1'])} "
                f"| {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |"
            )

    # McNemar's pairwise comparisons
    cond_names = sorted(all_predictions.keys())
    if len(cond_names) >= 2:
        lines.append("\n## Pairwise Comparison (McNemar's test)\n")
        lines.append("| Pair | Discordant (b, c) | p-value | Significant? |")
        lines.append("|------|:-----------------:|:-------:|:------------:|")

        primary_indices = [i for i, g in gt.items() if g in ("correct", "incorrect")]

        for i in range(len(cond_names)):
            for j in range(i + 1, len(cond_names)):
                a_name, b_name = cond_names[i], cond_names[j]
                a_preds, b_preds = all_predictions[a_name], all_predictions[b_name]

                b_disc = 0  # a correct, b wrong
                c_disc = 0  # a wrong, b correct

                for idx in primary_indices:
                    truth = gt[idx]
                    expected = "PASS" if truth == "correct" else "FLAGGED"

                    a_binary = "PASS" if a_preds.get(idx, "?") == "PASS" else "FLAGGED"
                    b_binary = "PASS" if b_preds.get(idx, "?") == "PASS" else "FLAGGED"

                    a_ok = (a_binary == expected)
                    b_ok = (b_binary == expected)

                    if a_ok and not b_ok:
                        b_disc += 1
                    elif b_ok and not a_ok:
                        c_disc += 1

                p = mcnemar_p(b_disc, c_disc)
                sig = "Yes" if p < 0.05 else "No"
                lines.append(f"| {a_name} vs {b_name} | ({b_disc}, {c_disc}) "
                             f"| {p:.4f} | {sig} |")

    report = "\n".join(lines) + "\n"
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved: {args.output}", flush=True)

    # Also save as JSON
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": [{k: v for k, v in m.items()} for m in all_metrics],
            "ideal_metrics": [{k: v for k, v in m.items()} for m in ideal_metrics],
            "ground_truth_distribution": dict(gt_dist),
            "conditions_evaluated": cond_names,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
