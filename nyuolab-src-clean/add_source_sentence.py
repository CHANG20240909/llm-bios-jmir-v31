"""
Add source_sentence column to CSV phrase files.

Two modes:
  auto   — 1:1 mapping: split txt by blank lines, match to CSV rows by order
  manual — JSON mapping: use a JSON file that maps row indices to source text

Usage:
    python add_source_sentence.py auto \
        --txt ../llm_answer_easy.txt \
        --csv ../data/llm_answer_easy_phrases.csv

    python add_source_sentence.py manual \
        --map ../data/llm_answer_source_map.json \
        --csv ../data/phase2a5_llm_answer_phrases.csv
"""

import argparse
import json
import sys

import pandas as pd


def split_sentences(txt_path):
    """Split a text file into sentences separated by blank lines."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    return blocks


def auto_map(txt_path, csv_path):
    """1:1 mapping between text blocks and CSV rows."""
    sentences = split_sentences(txt_path)
    df = pd.read_csv(csv_path)

    if len(sentences) != len(df):
        print(f"[WARN] Sentence count ({len(sentences)}) != CSV row count ({len(df)})")
        print("  Mapping as many as possible...")
        n = min(len(sentences), len(df))
    else:
        n = len(df)

    if "source_sentence" not in df.columns:
        df["source_sentence"] = ""

    for i in range(n):
        df.loc[i, "source_sentence"] = sentences[i]

    cols = ["source_sentence"] + [c for c in df.columns if c != "source_sentence"]
    df = df[cols]

    df.to_csv(csv_path, index=False)
    print(f"[OK] Added source_sentence to {csv_path} ({n} rows)")


def manual_map(map_path, csv_path):
    """JSON-based manual mapping for narrative texts."""
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    df = pd.read_csv(csv_path)

    if "source_sentence" not in df.columns:
        df["source_sentence"] = ""

    for entry in mapping:
        idx = entry["row_index"]
        src = entry["source_sentence"]
        if idx < len(df):
            df.loc[idx, "source_sentence"] = src

    cols = ["source_sentence"] + [c for c in df.columns if c != "source_sentence"]
    df = df[cols]

    df.to_csv(csv_path, index=False)
    print(f"[OK] Added source_sentence to {csv_path} ({len(mapping)} mappings)")


def main():
    parser = argparse.ArgumentParser(description="Add source_sentence column to CSV")
    subparsers = parser.add_subparsers(dest="mode")

    auto_p = subparsers.add_parser("auto", help="1:1 auto mapping from txt")
    auto_p.add_argument("--txt", required=True, help="Source text file")
    auto_p.add_argument("--csv", required=True, help="Target CSV file")

    manual_p = subparsers.add_parser("manual", help="Manual JSON mapping")
    manual_p.add_argument("--map", required=True, help="JSON mapping file")
    manual_p.add_argument("--csv", required=True, help="Target CSV file")

    args = parser.parse_args()

    if args.mode == "auto":
        auto_map(args.txt, args.csv)
    elif args.mode == "manual":
        manual_map(args.map, args.csv)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
