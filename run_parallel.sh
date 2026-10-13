#!/bin/bash
# Run every rater as its own process (separate providers, separate logs). Usage: ./run_parallel.sh [--pilot]
cd "$(dirname "$0")"
PY=../pvb_env/bin/python
mkdir -p outputs/logs
TAG=${1:+pilot}; TAG=${TAG:-full}
$PY 06_rate_openai.py $1 > outputs/logs/${TAG}_gpt-4.1.log 2>&1 &
for m in gemma-4-31b qwen3-235b deepseek-v4-pro solar-pro-4; do
  $PY 07_rate_openrouter.py --model $m $1 > outputs/logs/${TAG}_${m}.log 2>&1 &
done
wait
echo "all raters finished"
