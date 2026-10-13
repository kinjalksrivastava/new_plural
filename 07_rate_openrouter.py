"""
===============================================================================
PluralValueBench v2 — Stage II, Step 7: OpenRouter raters (async)
===============================================================================

All non-GPT raters via OpenRouter (config.OPENROUTER_API_KEY). For each model
the provider is pinned without fallbacks; hybrid models are called with
reasoning disabled, and the run aborts if any reply reports reasoning tokens.
Prompt, parser and logging are shared with the GPT rater (stage2_common.py).

Usage:
  python 07_rate_openrouter.py --model deepseek-v4-pro --pilot
  python 07_rate_openrouter.py --model all --pilot
  python 07_rate_openrouter.py --model all          # full run (resumable)
===============================================================================
"""

import argparse
import asyncio
import json

import config
import stage2_common as sc


def make_payload_fn(spec: dict):
    def make_payload(messages, k):
        p = {
            "model": spec["model"],
            "messages": messages,
            "temperature": config.RATER_TEMPERATURE,
            "max_tokens": config.RATER_MAX_TOKENS,
            "response_format": sc.response_format(k),
            "provider": {**spec["provider"], "require_parameters": True},
            "usage": {"include": True},
        }
        if spec["seed"]:
            p["seed"] = config.RATER_SEED
        if spec["disable_reasoning"]:
            p["reasoning"] = {"enabled": False}
        return p
    return make_payload


def run_model(key: str, pilot: bool, limit):
    spec = config.OPENROUTER_RATERS[key]
    rows = sc.select_rows(pilot, limit)
    todo = sc.todo_rows(rows, key)
    print(f"\n  {key} ({spec['model']} @ {spec['provider']['order']}): "
          f"{len(rows):,} rows | to do: {len(todo):,}")
    if len(todo):
        headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                   "Content-Type": "application/json", "X-Title": "PluralValueBench"}
        stats = asyncio.run(sc.run_all(todo, key, config.OPENROUTER_URL, headers,
                                       make_payload_fn(spec),
                                       n_workers=spec.get("workers", config.N_ASYNC_WORKERS)))
        if stats["reasoning_tokens"] > 0:
            raise SystemExit(f"  {key}: {stats['reasoning_tokens']} reasoning tokens reported "
                             "although reasoning must be off. Stopping.")
    summary = sc.summarize(rows, key)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.OPENROUTER_RATERS) + ["all"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is empty in config.py")
    sc.self_test()
    keys = list(config.OPENROUTER_RATERS) if args.model == "all" else [args.model]
    for key in keys:
        run_model(key, args.pilot, args.limit)


if __name__ == "__main__":
    main()
