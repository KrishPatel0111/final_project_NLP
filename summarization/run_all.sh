#!/usr/bin/env bash
set -e

python summarization/run_textrank.py
python summarization/run_bart.py
python summarization/run_pegasus.py
python summarization/run_flan.py
python summarization/run_mt5.py
# uncomment if you have GPU:
# python summarization/mistral_runner.py

python summarization/merge_outputs.py
