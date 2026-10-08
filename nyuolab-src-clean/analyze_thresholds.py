"""
Threshold analysis tool for Phase 1 alarm system.

Runs verify_medical_phrase in alarm_mode over a dataset, collects per-component
distances, and evaluates PASS/WARN/FAIL classification across a grid of
(strict_threshold, loose_threshold) pairs.

Usage:
    python analyze_thresholds.py --config ../configs/phase1_eval50_alarm.yaml
    python analyze_thresholds.py --config ../configs/phase1_eval50_mixed_alarm.yaml \
        --save-distances ../outputs/distances_eval50_mixed.csv
"""

import argparse
import sys

import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from utils import MedicalPhrase, classify_distance, set_seeds
from screen_outputs import setup_kg_vector_db, verify_medical_phrase


def collect_raw_distances(config, vector_db, knowledge_graph, df):
    """Run alarm-mode verification and collect per-component distances."""
    rows = []
    for idx in tqdm(range(len(df)), desc="Collecting distances"):
        row = df.iloc[idx]
        phrase = MedicalPhrase(row["origin"], row["relationship"], row["target"])

        vr = verify_medical_phrase(
            vector_db, knowledge_graph, phrase,
            k_r=config.verify.k_r,
            k_c=config.verify.k_c,
            check_negatives=config.verify.check_negatives,
            check_exact_relationships=config.verify.check_exact_relationships,
            check_exact_concepts=config.verify.check_exact_concepts,
            alarm_mode=True,
            n_candidates=config.get("n_candidates", 3),
            strict_threshold=0.05,
            loose_threshold=0.3,
        )

        gt = row.get("gt_result", row.get("result", None))
        rows.append({
            "origin": row["origin"],
            "relationship": row["relationship"],
            "target": row["target"],
            "gt_result": gt,
            "path_exists": vr.path_exists,
            "rel_distance": vr.relationship.distance,
            "rel_matched": vr.relationship.matched_text,
            "origin_distance": vr.origin.distance,
            "origin_matched": vr.origin.matched_text,
            "target_distance": vr.target.distance,
            "target_matched": vr.target.matched_text,
        })

    return pd.DataFrame(rows)


def evaluate_thresholds(dist_df):
    """Grid search over threshold pairs and compute metrics."""
    strict_vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    loose_vals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

    has_gt = dist_df["gt_result"].notna().all()

    if has_gt:
        dist_df["gt_bool"] = dist_df["gt_result"].apply(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
        )

    results = []
    for strict in strict_vals:
        for loose in loose_vals:
            if strict >= loose:
                continue

            alarm_flags = []
            for _, row in dist_df.iterrows():
                if row["path_exists"]:
                    alarm_flags.append(False)
                else:
                    statuses = [
                        classify_distance(row["rel_distance"], strict, loose),
                        classify_distance(row["origin_distance"], strict, loose),
                        classify_distance(row["target_distance"], strict, loose),
                    ]
                    alarm_flags.append(True)

            dist_df["alarm"] = alarm_flags

            n_alarm = sum(alarm_flags)
            n_pass = len(alarm_flags) - n_alarm
            path_exists_count = int(dist_df["path_exists"].sum())

            entry = {
                "strict": strict,
                "loose": loose,
                "path_exists": path_exists_count,
                "alarm": n_alarm,
                "pass": n_pass,
            }

            if has_gt:
                tp = int(((dist_df["alarm"]) & (~dist_df["gt_bool"])).sum())
                fp = int(((dist_df["alarm"]) & (dist_df["gt_bool"])).sum())
                fn = int(((~dist_df["alarm"]) & (~dist_df["gt_bool"])).sum())
                tn = int(((~dist_df["alarm"]) & (dist_df["gt_bool"])).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                far = fp / (fp + tn) if (fp + tn) > 0 else 0

                entry.update({
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "Precision": round(precision * 100, 1),
                    "Recall": round(recall * 100, 1),
                    "F1": round(f1 * 100, 1),
                    "FAR": round(far * 100, 1),
                })

            results.append(entry)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 threshold analysis")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--save-distances", default=None, help="Save raw distances CSV")
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)
    set_seeds(config.seed)

    knowledge_graph, vector_db = setup_kg_vector_db(config)
    df = pd.read_csv(config.medical_phrases)

    print(f"\nDataset: {config.medical_phrases}")
    print(f"  rows: {len(df)}")

    dist_df = collect_raw_distances(config, vector_db, knowledge_graph, df)

    if args.save_distances:
        dist_df.to_csv(args.save_distances, index=False)
        print(f"\nDistances saved to: {args.save_distances}")

    print(f"\n--- Distance Statistics ---")
    print(f"  path_exists = {int(dist_df['path_exists'].sum())}/{len(dist_df)}")
    for col in ["rel_distance", "origin_distance", "target_distance"]:
        vals = dist_df[col]
        print(f"  {col}: mean={vals.mean():.4f}  median={vals.median():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    has_gt = dist_df["gt_result"].notna().all()
    if not has_gt:
        print("\n  no ground-truth labels → False Alarm mode only")
    else:
        print(f"\n  ground-truth present → computing Precision/Recall/F1")

    grid_df = evaluate_thresholds(dist_df)

    print(f"\n--- Threshold Grid Results ({len(grid_df)} combinations) ---")
    if has_gt:
        grid_df_sorted = grid_df.sort_values("F1", ascending=False)
        print(grid_df_sorted.to_string(index=False))
        best = grid_df_sorted.iloc[0]
        print(f"\n  BEST: strict={best['strict']}, loose={best['loose']}, "
              f"F1={best['F1']}%, Precision={best['Precision']}%, "
              f"Recall={best['Recall']}%, FAR={best['FAR']}%")
    else:
        print(grid_df.to_string(index=False))


if __name__ == "__main__":
    main()
