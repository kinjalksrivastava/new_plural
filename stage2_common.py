"""
PluralValueBench v2 — Stage II shared rating engine.

Used by 06_rate_openai.py (GPT) and 07_rate_openrouter.py (all other models).

Elicitation: a single neutral prompt asking for the full answer distribution
of the country's adult population (verbalised distribution; Meister et al.,
2025). No anchoring context (no GDP / religiosity / region text), no
"do not be certain" instruction, no probability floor. The model's numbers are
used as returned; the only post-processing is renormalising an output whose
sum lies within config.DIST_SUM_ACCEPT. Anything else is a failure (one
repair turn), never patched.

Every reply is appended to outputs/stage2/<model_key>.jsonl. A logged result is
reused only if its prompt hash matches the current prompt.
"""

import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter

import httpx
import pandas as pd

import config

SYSTEM_PROMPT = ("You are an expert in cross-national public opinion research. "
                 "Respond with JSON only.")


# =============================================================================
# PROMPT + SCHEMA
# =============================================================================

def build_prompt(question_text: str, options: list, country: str) -> str:
    opts = "\n".join(f"{i}: {o}" for i, o in enumerate(options))
    keys = ", ".join(f'"{i}": p{i}' for i in range(len(options)))
    return (
        "Below is a question from a nationally representative public opinion survey "
        f"conducted in {country}. Estimate the share of adults in {country} who chose "
        "each answer option.\n\n"
        f'Question: "{question_text}"\n'
        f"Options:\n{opts}\n\n"
        "Return a JSON object that maps every option number to the estimated share "
        "(a number between 0 and 1); the shares must sum to 1:\n"
        "{" + keys + "}"
    )


def response_format(k: int) -> dict:
    keys = [str(i) for i in range(k)]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_distribution",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: {"type": "number"} for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


def repair_instruction(k: int) -> str:
    return ("Your previous reply was not valid. Reply with ONLY a JSON object with the "
            f"keys " + ", ".join(f'"{i}"' for i in range(k)) +
            ", each mapped to a number between 0 and 1, summing to 1.")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + prompt).encode()).hexdigest()


# =============================================================================
# PARSING (strict)
# =============================================================================

def _first_json_object(text: str):
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:j + 1]
        start = text.find("{", start + 1)
    return None


def parse_distribution(text: str, k: int):
    """-> (list of k floats summing to 1, raw_sum) or (None, error)."""
    blob = _first_json_object(text or "")
    if blob is None:
        return None, "no JSON object"
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "not an object"
    keys = [str(i) for i in range(k)]
    if sorted(obj.keys(), key=str) != sorted(keys, key=str):
        return None, f"keys {sorted(obj.keys())} != expected 0..{k-1}"
    vals = []
    for key in keys:
        v = obj[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"non-numeric value for option {key}: {v!r}"
        if v < 0 or v > 1:
            return None, f"value out of [0,1] for option {key}: {v}"
        vals.append(float(v))
    s = sum(vals)
    lo, hi = config.DIST_SUM_ACCEPT
    if not lo <= s <= hi:
        return None, f"sum {s:.4f} outside {config.DIST_SUM_ACCEPT}"
    return [v / s for v in vals], s


def self_test():
    p, s = parse_distribution('```json\n{"0": 0.5, "1": 0.3, "2": 0.2}\n```', 3)
    assert p is not None and abs(sum(p) - 1) < 1e-9
    assert parse_distribution('{"0": 0.5, "1": 0.5}', 3)[0] is None              # missing key
    assert parse_distribution('{"0": 0.5, "1": 0.3, "2": 0.4}', 3)[0] is None    # sum 1.2
    assert parse_distribution('{"0": 0.0, "1": 1.0, "2": 0.0}', 3)[0] == [0.0, 1.0, 0.0]  # zeros kept
    assert parse_distribution('{"0": -0.1, "1": 0.6, "2": 0.5}', 3)[0] is None
    assert parse_distribution('{"0": 50, "1": 30, "2": 20}', 3)[0] is None       # percentages rejected


# =============================================================================
# LOG / CHECKPOINT
# =============================================================================

def log_path(model_key: str):
    return config.STAGE2_OUT_DIR / f"{model_key}.jsonl"


def load_log(model_key: str) -> dict:
    latest = {}
    p = log_path(model_key)
    if p.exists():
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    latest[(r["question_id"], r["country"])] = r
    return latest


def append_log(model_key: str, rec: dict):
    config.STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path(model_key), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_done(rec: dict, prompt: str) -> bool:
    return rec.get("status") == "ok" and rec.get("prompt_sha256") == prompt_hash(prompt)


# =============================================================================
# ASYNC ENGINE
# =============================================================================

class FatalAPIError(Exception):
    pass


async def post_with_retry(client, url, payload):
    last = None
    for attempt in range(config.RATER_MAX_ATTEMPTS):
        try:
            r = await client.post(url, json=payload)
            if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            elif r.status_code != 200:
                raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:600]}")
            else:
                data = r.json()
                if "error" in data:
                    last = f"API error: {str(data['error'])[:300]}"
                elif not data.get("choices"):
                    last = "empty choices"
                else:
                    return data
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = f"{type(e).__name__}: {e}"
        await asyncio.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"gave up after {config.RATER_MAX_ATTEMPTS} attempts: {last}")


def reasoning_tokens(usage: dict) -> int:
    return int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)


async def rate_one(client, sem, url, make_payload, model_key, row, stats):
    options = json.loads(row.options)
    k = len(options)
    prompt = build_prompt(row.question_text, options, row.country)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    rec = {"question_id": row.question_id, "country": row.country, "k": k,
           "prompt_sha256": prompt_hash(prompt), "raw_output": None, "repair_output": None,
           "provider": None, "model_returned": None, "usage": [], "latency_s": [],
           "distribution": None, "raw_sum": None, "status": "failed", "error": None}
    async with sem:
        try:
            t0 = time.perf_counter()
            data = await post_with_retry(client, url, make_payload(messages, k))
            rec["latency_s"].append(round(time.perf_counter() - t0, 3))
            text = data["choices"][0]["message"].get("content") or ""
            rec.update(raw_output=text, provider=data.get("provider"), model_returned=data.get("model"))
            rec["usage"].append(data.get("usage", {}))
            dist, info = parse_distribution(text, k)
            if dist is None:
                t0 = time.perf_counter()
                data2 = await post_with_retry(client, url, make_payload(messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": repair_instruction(k)}], k))
                rec["latency_s"].append(round(time.perf_counter() - t0, 3))
                text2 = data2["choices"][0]["message"].get("content") or ""
                rec["repair_output"] = text2
                rec["usage"].append(data2.get("usage", {}))
                dist, info = parse_distribution(text2, k)
            if dist is not None:
                rec.update(distribution=dist, raw_sum=info, status="ok")
            else:
                rec["error"] = info
        except FatalAPIError:
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    rec["reasoning_tokens"] = sum(reasoning_tokens(u) for u in rec["usage"])
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_log(model_key, rec)
    stats[rec["status"]] += 1
    stats["reasoning_tokens"] += rec["reasoning_tokens"]


async def run_all(rows: pd.DataFrame, model_key: str, url: str, headers: dict, make_payload,
                  n_workers: int = config.N_ASYNC_WORKERS):
    from tqdm import tqdm
    sem = asyncio.Semaphore(n_workers)
    stats = Counter()
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(120.0)) as client:
        tasks = [asyncio.create_task(rate_one(client, sem, url, make_payload, model_key, r, stats))
                 for r in rows.itertuples()]
        with tqdm(total=len(tasks), desc=f"  {model_key}") as bar:
            for fut in asyncio.as_completed(tasks):
                try:
                    await fut
                except FatalAPIError as e:
                    for t in tasks:
                        t.cancel()
                    raise SystemExit(f"\n  Fatal API error, stopping: {e}")
                bar.update(1)
                bar.set_postfix(ok=stats["ok"], failed=stats["failed"])
    return stats


# =============================================================================
# DRIVER (shared by both rater scripts)
# =============================================================================

def select_rows(pilot: bool, limit: int | None) -> pd.DataFrame:
    rows = pd.read_csv(config.STAGE2_INPUT_CSV)
    rows = rows[~rows.structural_zero]
    if pilot:
        rows = rows[rows.is_pilot]
    if limit:
        rows = rows.head(limit)
    return rows.reset_index(drop=True)


def todo_rows(rows: pd.DataFrame, model_key: str) -> pd.DataFrame:
    log = load_log(model_key)
    keep = []
    for r in rows.itertuples():
        prompt = build_prompt(r.question_text, json.loads(r.options), r.country)
        keep.append(not is_done(log.get((r.question_id, r.country), {}), prompt))
    return rows[keep]


def summarize(rows: pd.DataFrame, model_key: str) -> dict:
    log = load_log(model_key)
    recs = [log[(r.question_id, r.country)] for r in rows.itertuples()
            if (r.question_id, r.country) in log]
    ok = [r for r in recs if r["status"] == "ok"]
    usage = [u for r in recs for u in r["usage"]]
    lat = [x for r in recs for x in r["latency_s"][:1]]
    return {
        "model_key": model_key,
        "n_rows": int(len(rows)),
        "n_ok": len(ok),
        "n_failed": len(recs) - len(ok),
        "n_missing": int(len(rows) - len(recs)),
        "n_repaired": sum(1 for r in ok if r["repair_output"] is not None),
        "reasoning_tokens_total": sum(r.get("reasoning_tokens", 0) for r in recs),
        "providers": dict(Counter(r["provider"] for r in recs)),
        "tokens_in": sum(u.get("prompt_tokens", 0) for u in usage),
        "tokens_out": sum(u.get("completion_tokens", 0) for u in usage),
        "cost_usd_reported": round(sum(float(u.get("cost") or 0) for u in usage), 4),
        "latency_s_mean": round(sum(lat) / len(lat), 3) if lat else None,
        "error_examples": Counter(r["error"] for r in recs if r["status"] != "ok").most_common(5),
    }
