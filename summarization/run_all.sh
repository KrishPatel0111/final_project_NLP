#!/usr/bin/env bash
set -e

run_model() {
    model_name=$1
    script=$2
    echo "=========================================="
    echo "Running $model_name..."
    echo "=========================================="
    if python summarization/$script; then
        echo "✅ $model_name completed successfully"
    else
        echo "❌ $model_name failed!"
        exit 1
    fi
}

# TextRank already done
echo "[SKIP] TextRank (already generated)"
#run_model "TextRank" "run_textrank.py"
run_model "BART" "run_bart.py"
run_model "PEGASUS" "run_pegasus.py"
run_model "FLAN-T5" "run_flan.py"
run_model "mT5" "run_mt5.py"
run_model "Mistral" "run_mistral.py"

echo "=========================================="
echo "Merging outputs..."
python summarization/merge_outputs.py
echo "✅ Pipeline complete!"