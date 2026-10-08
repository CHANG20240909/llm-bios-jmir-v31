"""
Negative sample generator for mixed evaluation datasets.

Creates false triples from a positive-only eval set using three strategies:
  1. target_shuffle  — swap target with a random target from another row
  2. origin_shuffle  — swap origin with a random origin from another row
  3. relation_mismatch — assign a random different relationship

Usage:
    python generate_negatives.py \
        --input ../data/eval_50_from_200k.csv \
        --output ../data/eval_50_mixed.csv \
        --n-negatives 50 \
        --seed 0
"""

import argparse
import random

import pandas as pd


def generate_false_triples(df, n, seed=0):
    """Generate n false triples using three balanced strategies."""
    rng = random.Random(seed)

    origins = df["origin"].tolist()
    targets = df["target"].tolist()
    relationships = df["relationship"].unique().tolist()

    per_strategy = n // 3
    remainder = n - per_strategy * 3

    counts = {
        "target_shuffle": per_strategy + (1 if remainder > 0 else 0),
        "origin_shuffle": per_strategy + (1 if remainder > 1 else 0),
        "relation_mismatch": per_strategy,
    }

    negatives = []

    for strategy, count in counts.items():
        attempts = 0
        generated = 0
        while generated < count and attempts < count * 50:
            attempts += 1
            row = df.iloc[rng.randint(0, len(df) - 1)]

            if strategy == "target_shuffle":
                new_target = rng.choice(targets)
                if new_target == row["target"]:
                    continue
                neg = {
                    "origin": row["origin"],
                    "relationship": row["relationship"],
                    "target": new_target,
                    "gt_result": False,
                    "strategy": strategy,
                }
            elif strategy == "origin_shuffle":
                new_origin = rng.choice(origins)
                if new_origin == row["origin"]:
                    continue
                neg = {
                    "origin": new_origin,
                    "relationship": row["relationship"],
                    "target": row["target"],
                    "gt_result": False,
                    "strategy": strategy,
                }
            else:
                other_rels = [r for r in relationships if r != row["relationship"]]
                if not other_rels:
                    continue
                neg = {
                    "origin": row["origin"],
                    "relationship": rng.choice(other_rels),
                    "target": row["target"],
                    "gt_result": False,
                    "strategy": strategy,
                }

            dup = any(
                n["origin"] == neg["origin"]
                and n["relationship"] == neg["relationship"]
                and n["target"] == neg["target"]
                for n in negatives
            )
            if not dup:
                negatives.append(neg)
                generated += 1

    return pd.DataFrame(negatives)


def main():
    parser = argparse.ArgumentParser(description="Generate negative samples for mixed eval")
    parser.add_argument("--input", required=True, help="Positive-only eval CSV")
    parser.add_argument("--output", required=True, help="Output mixed CSV path")
    parser.add_argument("--n-negatives", type=int, default=50, help="Number of negatives (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    pos_df = pd.read_csv(args.input)
    pos_df["gt_result"] = True
    pos_df["strategy"] = "positive"

    print(f"Positives: {len(pos_df)}")

    neg_df = generate_false_triples(pos_df, args.n_negatives, args.seed)
    print(f"Negatives: {len(neg_df)}")

    strategy_counts = neg_df["strategy"].value_counts()
    for s, c in strategy_counts.items():
        print(f"  {s}: {c}")

    mixed = pd.concat([pos_df, neg_df], ignore_index=True)
    mixed = mixed.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    mixed.to_csv(args.output, index=False)
    print(f"\nMixed dataset saved: {args.output}  (total: {len(mixed)})")


if __name__ == "__main__":
    main()
