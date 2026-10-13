"""
===============================================================================
PluralValueBench v2 — Stage II, Step 6b: complete GPT-4.1 via OpenRouter
===============================================================================

The direct OpenAI run (06_rate_openai.py) stopped when the OpenAI account ran
out of credit. This script completes GPT-4.1 through OpenRouter, pinned to
OpenAI's own endpoint (provider "openai", no fallbacks — never Azure), so the
same model serves every row. Identical prompt, decoding and parser
(stage2_common.py); results are appended to the same log
(outputs/stage2/gpt-4.1.jsonl), so rows already answered with the current
prompt are skipped and only remaining, failed or prompt-changed rows are sent.

Usage:
  python 06b_complete_gpt_openrouter.py --limit 20   # smoke test
  python 06b_complete_gpt_openrouter.py              # complete the run
===============================================================================
"""

import argparse
import asyncio
import json

import config
import stage2_common as sc

MODEL_KEY = "gpt-4.1"
OPENROUTER_MODEL = "openai/gpt-4.1"
PROVIDER = {"order": ["openai"], "allow_fallbacks": False, "require_parameters": True}


def make_payload(messages, k):
    return {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": config.RATER_TEMPERATURE,
        "seed": config.RATER_SEED,
        "max_tokens": config.RATER_MAX_TOKENS,
        "response_format": sc.response_format(k),
        "provider": PROVIDER,
        "usage": {"include": True},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not config.OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is empty in config.py")
    sc.self_test()

    rows = sc.select_rows(False, None)
    todo = sc.todo_rows(rows, MODEL_KEY)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"  {MODEL_KEY} via OpenRouter ({OPENROUTER_MODEL} @ OpenAI): "
          f"{len(rows):,} rows | to do: {len(todo):,}")
    if len(todo):
        headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                   "Content-Type": "application/json", "X-Title": "PluralValueBench"}
        stats = asyncio.run(sc.run_all(todo, MODEL_KEY, config.OPENROUTER_URL, headers, make_payload))
        if stats["reasoning_tokens"] > 0:
            raise SystemExit(f"  {stats['reasoning_tokens']} reasoning tokens reported. Stopping.")
    print(json.dumps(sc.summarize(rows, MODEL_KEY), indent=2))


if __name__ == "__main__":
    main()
