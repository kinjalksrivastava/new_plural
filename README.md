# PluralValueBench

Code, data and model predictions for the paper "Do Large Language Models Recognize Value Pluralism Across Cultures?".

PluralValueBench compares the answer distributions that large language models state for a country with real
survey distributions. Stage I builds a value-labeled corpus from GlobalOpinionQA. Stage II asks five models
for the answer distribution of each country and scores the predictions.

## Contents

| Path | What it holds |
|---|---|
| `01_build_dataset.py` to `05_prepare_stage2.py` | Stage I: cleaning, value labels, WVS Wave 6 items, duplicate removal, country metadata |
| `06_rate_openai.py`, `06b_complete_gpt_openrouter.py`, `07_rate_openrouter.py` | Stage II: distribution prediction with the five models |
| `08_metrics.py` | All metrics, bootstrap confidence intervals and tests |
| `09_human_validation_sample.py`, `10_human_validation_agreement.py` | Human validation sample and agreement statistics |
| `codebook.py`, `stage2_common.py`, `config.py` | Value definitions, prompt and parser, settings |
| `data/Q3_final_rows.csv`, `data/Q3_final_questions.csv` | Final corpus: 42,375 question and country pairs, 2,482 questions with value labels |
| `data/stage2_input.csv` | The evaluated pairs with Schwartz regions and orientation scores |
| `data/*_report.json` | Counts for every cleaning step |
| `data/raw/schwartz/` | Schwartz cultural orientation scores with their source note |
| `predictions/*.jsonl.gz` | Every model request and response (one file per model) and the value annotation log |
| `results/full/metrics.json`, `results/full/row_metrics.csv` | All reported numbers and the per-pair metrics |
| `human_validation/` | The 150 validation questions, the two raters' labels and the model's labels (as CSV), and the agreement report |

## Reproducing the numbers from the released predictions

```bash
pip install -r requirements.txt
./unpack_predictions.sh        # restores the logs to outputs/
python 08_metrics.py           # writes results/full/metrics.json and row_metrics.csv
```

No API key is needed for this step.

## Re-running the pipeline

1. Download GlobalOpinionQA (`global_opinions.csv`) to `data/raw/`. The exact revision and checksum are in `config.py`.
2. Download the World Values Survey Wave 6 data file from the WVS website to `data/raw/wvs6/` (registration required). We do not redistribute it.
3. Set the keys in the environment. Keys are never stored in this repository.

```bash
export OPENROUTER_API_KEY=...   # value annotation and the four open models
export OPENAI_API_KEY=...       # GPT-4.1
```

4. Run the scripts in numerical order. `run_parallel.sh` starts the Stage II raters together.

## Data sources and terms

- GlobalOpinionQA (Durmus et al.) is released under CC BY-NC-SA 4.0. The corpus files in `data/` are derived from it and are shared under the same license.
- World Values Survey Wave 6: only aggregate country distributions for two items are included, in line with the WVS conditions of use. The respondent-level file is not included.
- Schwartz cultural value orientation scores: see `data/raw/schwartz/SOURCE.md`.
