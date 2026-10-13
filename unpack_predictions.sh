#!/bin/bash
# Restores the model logs to the paths the scripts expect (outputs/).
set -e
mkdir -p outputs/stage2
for f in predictions/*.jsonl.gz; do
  name=$(basename "$f" .gz)
  if [ "$name" = "classifier_raw.jsonl" ]; then gunzip -c "$f" > "outputs/$name"; else gunzip -c "$f" > "outputs/stage2/$name"; fi
done
echo "logs restored to outputs/"
