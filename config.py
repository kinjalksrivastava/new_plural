"""
PluralValueBench v2 — central configuration.

API keys are read from the environment variables OPENAI_API_KEY and
OPENROUTER_API_KEY. Do not write keys into this file.
"""

import os
from pathlib import Path

# =============================================================================
# API KEYS  (from the environment)
# =============================================================================
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")   # used only by the GPT rater file
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")   # used by all non-GPT raters
HF_TOKEN           = os.environ.get("HF_TOKEN", "")   # optional; Mistral-7B-Instruct-v0.3 is not gated

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
OUTPUT_DIR  = ROOT_DIR / "outputs"

# Stage I, Step 1 (01_build_dataset.py)
RAW_GOQA_CSV       = RAW_DIR / "global_opinions.csv"
Q0_ROWS_CSV        = DATA_DIR / "Q0_rows.csv"         # one row per (question, country)
Q0_QUESTIONS_CSV   = DATA_DIR / "Q0_questions.csv"    # one row per question
Q0_REPORT_JSON     = DATA_DIR / "Q0_report.json"

# Stage I, Step 3 expansion source: WVS Wave 6 (v20201117), Schwartz battery
WVS6_DIR           = RAW_DIR / "wvs6"
WVS6_DATA_CSV      = WVS6_DIR / "WV6_Data_csv_v20201117.csv"   # ';'-separated, UTF-8 with BOM
WVS6_DATA_SHA256   = "a302c05a2792e82f54e32e0ae20ac8fa4e8cf21a72282fb2af962dd86a602146"
WVS6_COUNTRY_COL   = "V2"
WVS6_ISO3_COL      = "B_COUNTRY_ALPHA"
WVS6_WEIGHT_COL    = "V258"

# Stage I, Step 3 (03_expand_real_items.py)
EXPANSION_MIN_QUESTIONS = 50     # a value with fewer unique questions is low-resource
WVS6_MIN_VALID_N        = 100    # min. respondents with a valid answer per country
Q2_EXPANSION_ROWS_CSV   = DATA_DIR / "Q2_expansion_rows.csv"
Q2_ROWS_CSV             = DATA_DIR / "Q2_rows.csv"
Q2_REPORT_JSON          = DATA_DIR / "Q2_report.json"

# Stage I, Step 4 (04_dedup_finalize.py)
SIMHASH_BITS          = 64
SIMHASH_SHINGLE       = 3                 # word 3-gram shingles
SIMHASH_TAU           = 10                # Hamming threshold (as in Demo-SafetyBench)
SIMHASH_TAU_SWEEP     = [3, 6, 10, 15]    # sensitivity analysis
Q3_ROWS_CSV           = DATA_DIR / "Q3_final_rows.csv"       # FINAL Stage I corpus
Q3_QUESTIONS_CSV      = DATA_DIR / "Q3_final_questions.csv"
Q3_DEDUP_PAIRS_CSV    = DATA_DIR / "Q3_simhash_candidates.csv"
Q3_REPORT_JSON        = DATA_DIR / "Q3_report.json"
Q3_TABLE_TEX          = DATA_DIR / "Q3_table_stage1.tex"

# Stage I, Step 2 (02_classify_schwartz.py)
CLASSIFIER_RAW_JSONL   = OUTPUT_DIR / "classifier_raw.jsonl"   # append-only log / checkpoint
Q1_QUESTIONS_CSV       = DATA_DIR / "Q1_questions_classified.csv"
Q1_ROWS_CSV            = DATA_DIR / "Q1_rows.csv"
Q1_REPORT_JSON         = DATA_DIR / "Q1_report.json"

# =============================================================================
# SOURCE DATA  (GlobalOpinionQA; Durmus et al., 2023)
# Pinned to a fixed Hugging Face revision for reproducibility.
# =============================================================================
GOQA_REVISION = "cb2880488749218abb81802a94c2c62ebfde2f35"
GOQA_URL = (
    "https://huggingface.co/datasets/Anthropic/llm_global_opinions/resolve/"
    f"{GOQA_REVISION}/data/global_opinions.csv"
)

# Distributions whose raw sum deviates from 1 by more than this are flagged
# (they are always renormalised; all-zero distributions are dropped).
DIST_SUM_TOLERANCE = 0.02

# =============================================================================
# STAGE I CLASSIFIER  (OpenRouter; binary per-value labels, Kiesel et al. 2022)
# Open-weights, non-reasoning, disjoint from all Stage II raters.
# Pinned to one full-precision provider, no fallbacks, so every question is
# labelled by the identical deployment.
# =============================================================================
CLASSIFIER_MODEL_ID       = "meta-llama/llama-3.3-70b-instruct"
CLASSIFIER_PROVIDER       = {
    "order": ["coreweave"],
    "quantizations": ["fp16"],
    "allow_fallbacks": False,
    "require_parameters": True,
}
CLASSIFIER_MAX_TOKENS     = 200
CLASSIFIER_TEMPERATURE    = 0.0
CLASSIFIER_SEED           = 42
CLASSIFIER_WORKERS        = 20
CLASSIFIER_MAX_ATTEMPTS   = 6      # per request (network / 429 / 5xx); backoff between
OPENROUTER_URL            = "https://openrouter.ai/api/v1/chat/completions"

SCHWARTZ_VALUES = [
    "Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power",
    "Security", "Conformity", "Tradition", "Benevolence", "Universalism",
]

# =============================================================================
# STAGE II RATERS  (all non-reasoning; reasoning explicitly disabled for hybrids)
# Stage I classifier (Llama) is deliberately NOT among the raters.
# =============================================================================
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RATERS = {
    # key: (api model id, USD per 1M input, USD per 1M output)
    "gpt-4.1": ("gpt-4.1", 2.00, 8.00),
}
OPENROUTER_RATERS = {
    # key: model id, pinned provider (no fallbacks), reasoning switch, seed support
    "gemma-4-31b": {
        "model": "google/gemma-4-31b-it",
        "provider": {"order": ["deepinfra/fp8"], "allow_fallbacks": False},
        "disable_reasoning": True, "seed": True,
        "workers": 40,                                  # latency-bound (~6 s/call), no 429s observed
    },
    "qwen3-235b": {
        "model": "qwen/qwen3-235b-a22b-2507",
        "provider": {"order": ["deepinfra/fp8"], "allow_fallbacks": False},
        "disable_reasoning": False, "seed": True,       # instruct-only model
        "workers": 35,                                  # latency-bound (~6 s/call)
    },
    "deepseek-v4-pro": {
        "model": "deepseek/deepseek-v4-pro",
        "provider": {"order": ["streamlake/fp8"], "allow_fallbacks": False},
        "disable_reasoning": True, "seed": False,       # provider has no seed param
        "workers": 40,                                  # latency-bound, no 429s observed
    },
    "solar-pro-4": {
        "model": "upstage/solar-pro4",
        "provider": {"order": ["upstage"], "allow_fallbacks": False},
        "disable_reasoning": True, "seed": False,       # first-party only; no seed param
    },
}
# Dropped: mistral-large-2512 (first-party endpoint rate-limited at 20 workers;
# ~4 h for the full run at 5 workers). Spec kept for reference:
#     "mistral-large-3": {
#         "model": "mistralai/mistral-large-2512",
#         "provider": {"order": ["mistral/zdr"], "allow_fallbacks": False},
#         "disable_reasoning": False, "seed": True,       # non-reasoning model
#         "workers": 5,                                   # first-party endpoint rate-limits at 20
#     },
RATER_TEMPERATURE = 0.0
RATER_SEED        = 42
RATER_MAX_TOKENS  = 400
RATER_MAX_ATTEMPTS = 8
N_ASYNC_WORKERS   = 20
DIST_SUM_ACCEPT   = (0.90, 1.10)     # model output must sum into this range, then renormalised

# Stage II paths
SCHWARTZ_DIR          = RAW_DIR / "schwartz"
STAGE2_INPUT_CSV      = DATA_DIR / "stage2_input.csv"
STAGE2_PREP_REPORT    = DATA_DIR / "stage2_prep_report.json"
STAGE2_PILOT_IDS      = DATA_DIR / "stage2_pilot_question_ids.json"
STAGE2_OUT_DIR        = OUTPUT_DIR / "stage2"              # <model>.jsonl logs
RESULTS_DIR           = ROOT_DIR / "results"
PILOT_N_QUESTIONS     = 60
BOOTSTRAP_B           = 1000
