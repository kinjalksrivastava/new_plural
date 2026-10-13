"""
===============================================================================
PluralValueBench v2 — Stage II, Step 6: GPT rater (OpenAI API, async)
===============================================================================

GPT models only, called directly through the OpenAI API with
config.OPENAI_API_KEY. Non-reasoning model, temperature 0, seed 42,
strict JSON-schema output, config.N_ASYNC_WORKERS concurrent requests.
Prompt, parser and logging are shared with the OpenRouter raters
(stage2_common.py).

Usage:
  python 06_rate_openai.py --pilot              # pilot subset
  python 06_rate_openai.py --pilot --limit 20   # smoke test
  python 06_rate_openai.py                      # full run (resumable)
===============================================================================
"""

import argparse
import asyncio
import json

import config
import stage2_common as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1", choices=list(config.OPENAI_RATERS))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is empty in config.py")
    sc.self_test()
    model_id, usd_in, usd_out = config.OPENAI_RATERS[args.model]

    def make_payload(messages, k):
        return {
            "model": model_id,
            "messages": messages,
            "temperature": config.RATER_TEMPERATURE,
            "seed": config.RATER_SEED,
            "max_tokens": config.RATER_MAX_TOKENS,
            "response_format": sc.response_format(k),
        }

    rows = sc.select_rows(args.pilot, args.limit)
    todo = sc.todo_rows(rows, args.model)
    print(f"  {args.model}: {len(rows):,} rows | to do: {len(todo):,}")
    if len(todo):
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}",
                   "Content-Type": "application/json"}
        asyncio.run(sc.run_all(todo, args.model, config.OPENAI_URL, headers, make_payload))

    summary = sc.summarize(rows, args.model)
    summary["cost_usd_estimated"] = round(summary["tokens_in"] / 1e6 * usd_in
                                          + summary["tokens_out"] / 1e6 * usd_out, 4)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
