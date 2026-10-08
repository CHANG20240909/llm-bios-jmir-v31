#!/bin/bash
# ============================================================
# benchmark_500 C0-C4 Ablation Study
# ============================================================
# 실행 방법 (GCP):
#   1) git pull                                       # 최신 코드 동기화
#   2) tmux new -s ab500                              # SSH 끊김 방지
#   3) bash ~/llm-bios/tools/run_ablation_500.sh      # 전체 실행
#
# 개별 condition만 실행:
#   bash ~/llm-bios/tools/run_ablation_500.sh C1
#   bash ~/llm-bios/tools/run_ablation_500.sh C0
# ============================================================
set -euo pipefail

REPO_ROOT="$HOME/llm-bios"
cd "$REPO_ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

PHRASES="data/benchmark_500_phrases.csv"
OUT="$REPO_ROOT/outputs/ablation_500"
mkdir -p "$OUT"

ONLY="${1:-ALL}"

if [ ! -f "$PHRASES" ]; then
    echo "ERROR: $PHRASES not found. Run 'git pull' first." >&2
    exit 1
fi

if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
    echo "  .env loaded from $REPO_ROOT/.env"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set. C0 and C4 (GPT-4o) will fail."
    echo "  Option 1: export OPENAI_API_KEY='sk-...'"
    echo "  Option 2: echo 'OPENAI_API_KEY=sk-...' > ~/llm-bios/.env"
fi

echo "=== benchmark_500 Ablation (C0-C4) ==="
echo "  Phrases : $PHRASES ($(wc -l < "$PHRASES") lines)"
echo "  Output  : $OUT"
echo "  Run     : $ONLY"
echo ""

# ── Helper: run screen_outputs for a condition ──────────────
run_phase1() {
    local COND="$1"
    local CONFIG_SRC="$2"

    echo "────────────────────────────────────────"
    echo " [$COND] Phase 1"
    echo "────────────────────────────────────────"

    local TMP="/tmp/ablation_${COND}_b500.yaml"
    sed "s|REPO_ROOT|$REPO_ROOT|g; s|PLACEHOLDER_PHRASES.csv|benchmark_500_phrases.csv|g" \
        "$REPO_ROOT/configs/$CONFIG_SRC" > "$TMP"

    cd "$REPO_ROOT/nyuolab-src-clean"
    local T0=$(date +%s)
    python screen_outputs.py --config "$TMP" 2>&1 | tee "$OUT/${COND}_benchmark500.log"
    local T1=$(date +%s)
    echo "  [$COND] elapsed: $((T1 - T0))s"

    local SCORED="$REPO_ROOT/data/benchmark_500_phrases_scored.csv"
    if [ -f "$SCORED" ]; then
        cp "$SCORED" "$OUT/${COND}_benchmark500_scored.csv"
        echo "  [$COND] saved: $OUT/${COND}_benchmark500_scored.csv"
    else
        echo "  [$COND] WARNING: scored CSV not found at $SCORED"
    fi
    rm -f "$REPO_ROOT/data/benchmark_500_phrases_checkpoint.csv"
    echo ""
}

# ── C1: Exact matching ──────────────────────────────────────
if [ "$ONLY" = "ALL" ] || [ "$ONLY" = "C1" ]; then
    run_phase1 "C1" "ablation_C1_exact.yaml"
fi

# ── C2: Semantic matching ───────────────────────────────────
if [ "$ONLY" = "ALL" ] || [ "$ONLY" = "C2" ]; then
    run_phase1 "C2" "ablation_C2_semantic.yaml"
fi

# ── C3: Semantic + Modifier ─────────────────────────────────
if [ "$ONLY" = "ALL" ] || [ "$ONLY" = "C3" ]; then
    run_phase1 "C3" "ablation_C3_semantic_mod.yaml"
fi

# ── C4: C3 + Phase 2 LLM ───────────────────────────────────
if [ "$ONLY" = "ALL" ] || [ "$ONLY" = "C4" ]; then
    echo "────────────────────────────────────────"
    echo " [C4] Phase 2 (GPT-4o on C3 WARN/FAIL)"
    echo "────────────────────────────────────────"

    C3_SCORED="$OUT/C3_benchmark500_scored.csv"
    if [ ! -f "$C3_SCORED" ]; then
        echo "  ERROR: C3 scored not found at $C3_SCORED" >&2
        echo "  C4 requires C3. Run C3 first: bash $0 C3" >&2
        exit 1
    fi

    cat > /tmp/ablation_C4_b500_phase2.yaml <<YAML
phase2:
  scored_csv: ${C3_SCORED}
  output_jsonl: ${OUT}/C4_benchmark500_phase2.jsonl
  output_report: ${OUT}/C4_benchmark500_phase2_report.txt
  model: gpt-4o
YAML

    cd "$REPO_ROOT/nyuolab-src-clean"
    T0=$(date +%s)
    python phase2_pipeline.py --config /tmp/ablation_C4_b500_phase2.yaml \
        2>&1 | tee "$OUT/C4_benchmark500_phase2.log"
    T1=$(date +%s)
    echo "  [C4] Phase 2 elapsed: $((T1 - T0))s"

    python3 -c "
import csv, json

c3_path = '${C3_SCORED}'
p2_path = '${OUT}/C4_benchmark500_phase2.jsonl'
out_path = '${OUT}/C4_benchmark500_scored.csv'

with open(c3_path, encoding='utf-8') as f:
    phase1 = list(csv.DictReader(f))

p2_verdicts = {}
try:
    with open(p2_path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            key = (rec['origin'].lower().strip(),
                   rec['relationship'].lower().strip(),
                   rec['target'].lower().strip())
            p2_verdicts[key] = rec.get('llm_verdict', 'WARN')
except FileNotFoundError:
    pass

fieldnames = list(phase1[0].keys())
if 'phase2_verdict' not in fieldnames:
    fieldnames.append('phase2_verdict')

for row in phase1:
    key = (row['origin'].lower().strip(),
           row['relationship'].lower().strip(),
           row['target'].lower().strip())
    if key in p2_verdicts:
        row['alarm_level'] = p2_verdicts[key]
        row['phase2_verdict'] = p2_verdicts[key]
    else:
        row.setdefault('phase2_verdict', '')

with open(out_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(phase1)

print(f'  [C4] merged: {out_path} ({len(phase1)} rows, {len(p2_verdicts)} Phase 2 updates)')
"
    echo ""
fi

# ── C0: GPT-4o only (no KG) ────────────────────────────────
if [ "$ONLY" = "ALL" ] || [ "$ONLY" = "C0" ]; then
    echo "────────────────────────────────────────"
    echo " [C0] GPT-4o only baseline"
    echo "────────────────────────────────────────"

    cd "$REPO_ROOT"
    T0=$(date +%s)
    python tools/baseline_c0_gpt4o_only.py \
        --input "$PHRASES" \
        --output "$OUT/C0_benchmark500_scored.csv" \
        2>&1 | tee "$OUT/C0_benchmark500.log"
    T1=$(date +%s)
    echo "  [C0] elapsed: $((T1 - T0))s"
    echo ""
fi

# ── Metrics ─────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo " Computing metrics"
echo "════════════════════════════════════════"

cd "$REPO_ROOT"

C0_ARG=""
if [ -f "$OUT/C0_benchmark500_scored.csv" ]; then
    C0_ARG="--c0-csv $OUT/C0_benchmark500_scored.csv"
fi

python tools/compute_ablation_metrics.py \
    --ground-truth data/benchmark_500.csv \
    --results-dir "$OUT" \
    --output "$OUT/ablation_report_benchmark500.md" \
    $C0_ARG

echo ""
echo "════════════════════════════════════════"
echo " ALL DONE"
echo " Report: $OUT/ablation_report_benchmark500.md"
echo "════════════════════════════════════════"

echo ""
echo "--- Quick alarm distribution ---"
for f in "$OUT"/*_scored.csv; do
    [ -f "$f" ] || continue
    BNAME=$(basename "$f")
    echo "=== $BNAME ==="
    python3 -c "
import csv
from collections import Counter
with open('$f', encoding='utf-8') as fp:
    rows = list(csv.DictReader(fp))
c = Counter()
for r in rows:
    k = r.get('alarm_level', r.get('c0_verdict', r.get('result', '?')))
    c[k] += 1
for k in sorted(c):
    print(f'  {k}: {c[k]}/{len(rows)} ({c[k]/len(rows)*100:.0f}%)')
"
done
