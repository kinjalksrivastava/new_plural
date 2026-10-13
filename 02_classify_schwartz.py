"""
===============================================================================
PluralValueBench v2 — Stage I, Step 2: Schwartz value classification (Q1)
===============================================================================

Labels every question in Q0_questions.csv with Schwartz's ten basic values.

Protocol: binary yes/no judgement per value, multi-label (Kiesel et al., 2022;
Mirzakhmedova et al., 2024). A value is assigned iff the classifier answers
"yes". No probabilities, no threshold, no argmax fallback: a question with no
"yes" is labelled None and excluded from per-value analyses (as in
Demo-SafetyBench's None class).

Classifier: config.CLASSIFIER_MODEL_ID via OpenRouter, temperature 0, fixed
seed, pinned provider without fallbacks, JSON-schema constrained output,
config.CLASSIFIER_WORKERS concurrent async requests. Question text AND answer
options are shown — exactly what human annotators see (codebook.py).

Pipeline safeguards:
  - Strict parsing: all 10 values must be present as booleans; nothing is
    defaulted.
  - Transient errors (network, 429, 5xx) are retried with exponential backoff.
  - One repair turn if a reply does not parse.
  - Every reply is appended to outputs/classifier_raw.jsonl (raw text, provider,
    token usage). Re-running resumes: only questions without an "ok" record
    are sent again.
  - Q1 files are written only when every question has a valid label set
    (--allow_failures writes anyway, failures flagged).

Outputs:
  outputs/classifier_raw.jsonl        raw replies (log / checkpoint)
  data/Q1_questions_classified.csv    one row per question with labels
  data/Q1_rows.csv                    Q0_rows + labels
  data/Q1_report.json                 label statistics, cost, providers

Usage:
  python 02_classify_schwartz.py --dry_run      # print a prompt + parser tests
  python 02_classify_schwartz.py --limit 50     # pilot
  python 02_classify_schwartz.py                # full run (resumable)
===============================================================================
"""

import argparse
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
from codebook import codebook_text

VALUES = config.SCHWARTZ_VALUES
_KEY_LOOKUP = {re.sub(r"[^a-z]", "", v.lower()): v for v in VALUES}

SYSTEM_PROMPT = (
    "You are an expert annotator trained in Schwartz's Theory of Basic Human "
    "Values. You label survey questions. Respond with JSON only."
)

REPAIR_INSTRUCTION = (
    "Your previous reply was not in the required format. Reply with ONLY one "
    "JSON object containing exactly these ten keys: "
    + ", ".join(f'"{v}"' for v in VALUES)
    + ". Each value must be true or false. No other text."
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "schwartz_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {v: {"type": "boolean"} for v in VALUES},
            "required": VALUES,
            "additionalProperties": False,
        },
    },
}


# =============================================================================
# PROMPT
# =============================================================================

def build_prompt(question_text: str, options: list) -> str:
    options_block = "\n".join(f"  - {o}" for o in options)
    json_template = "{" + ", ".join(f'"{v}": true|false' for v in VALUES) + "}"
    return (
        "Label the survey question below with the Schwartz values that are at "
        "stake in it.\n\n"
        f"{codebook_text()}\n\n"
        f'Survey question: "{question_text}"\n'
        f"Answer options:\n{options_block}\n\n"
        "Reply with ONLY a JSON object with all ten keys (true = yes, "
        "false = no):\n"
        f"{json_template}"
    )


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


def parse_labels(text: str):
    """Parse a reply into {value: bool} for all 10 values -> (labels, error)."""
    blob = _first_json_object(text or "")
    if blob is None:
        return None, "no JSON object found"
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"

    out = {}
    for raw_key, raw_val in obj.items():
        key = _KEY_LOOKUP.get(re.sub(r"[^a-z]", "", str(raw_key).lower()))
        if key is None:
            return None, f"unknown key {raw_key!r}"
        if key in out:
            return None, f"duplicate key {key!r}"
        if isinstance(raw_val, str) and raw_val.strip().lower() in {"yes", "no", "true", "false"}:
            raw_val = raw_val.strip().lower() in {"yes", "true"}
        if not isinstance(raw_val, bool):
            return None, f"value for {key} is not a boolean: {raw_val!r}"
        out[key] = raw_val

    missing = [v for v in VALUES if v not in out]
    if missing:
        return None, f"missing keys: {missing}"
    return {v: out[v] for v in VALUES}, None


def prompt_hash(row) -> str:
    return hashlib.sha256(build_prompt(row.question_text, json.loads(row.options)).encode()).hexdigest()


def is_done(rec: dict, row) -> bool:
    """A logged label is reused only if it is OK AND was produced from the identical prompt."""
    return rec.get("status") == "ok" and rec.get("prompt_sha256") == prompt_hash(row)


def labels_list(judgements: dict) -> list:
    return [v for v in VALUES if judgements[v]]


# =============================================================================
# CHECKPOINT / LOG
# =============================================================================

def load_log() -> dict:
    """Latest record per question_id from the append-only JSONL log."""
    latest = {}
    if config.CLASSIFIER_RAW_JSONL.exists():
        with open(config.CLASSIFIER_RAW_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    latest[rec["question_id"]] = rec
    return latest


def append_log(rec: dict):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CLASSIFIER_RAW_JSONL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# OPENROUTER CALLS
# =============================================================================

class FatalAPIError(Exception):
    """Errors that retrying cannot fix (bad key, bad request, no provider)."""


async def call_openrouter(client: httpx.AsyncClient, messages: list) -> dict:
    payload = {
        "model":           config.CLASSIFIER_MODEL_ID,
        "messages":        messages,
        "temperature":     config.CLASSIFIER_TEMPERATURE,
        "seed":            config.CLASSIFIER_SEED,
        "max_tokens":      config.CLASSIFIER_MAX_TOKENS,
        "response_format": RESPONSE_FORMAT,
        "provider":        config.CLASSIFIER_PROVIDER,
    }
    last_err = None
    for attempt in range(config.CLASSIFIER_MAX_ATTEMPTS):
        try:
            resp = await client.post(config.OPENROUTER_URL, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504, 408):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            elif resp.status_code != 200:
                raise FatalAPIError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            else:
                data = resp.json()
                if "error" in data:
                    last_err = f"API error: {str(data['error'])[:300]}"
                elif not data.get("choices"):
                    last_err = "empty choices"
                else:
                    return {
                        "text":     data["choices"][0]["message"].get("content") or "",
                        "provider": data.get("provider"),
                        "model":    data.get("model"),
                        "usage":    data.get("usage", {}),
                    }
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"gave up after {config.CLASSIFIER_MAX_ATTEMPTS} attempts: {last_err}")


async def classify_one(client, sem, row, stats):
    prompt = build_prompt(row.question_text, json.loads(row.options))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    rec = {
        "question_id":   row.question_id,
        "model":         config.CLASSIFIER_MODEL_ID,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_output":    None,
        "repair_output": None,
        "provider":      None,
        "usage":         [],
        "labels":        None,
        "status":        "failed",
        "error":         None,
    }
    async with sem:
        try:
            first = await call_openrouter(client, messages)
            rec.update(raw_output=first["text"], provider=first["provider"])
            rec["usage"].append(first["usage"])
            labels, err = parse_labels(first["text"])
            if labels is None:
                repair = await call_openrouter(client, messages + [
                    {"role": "assistant", "content": first["text"]},
                    {"role": "user", "content": REPAIR_INSTRUCTION},
                ])
                rec["repair_output"] = repair["text"]
                rec["usage"].append(repair["usage"])
                labels, err = parse_labels(repair["text"])
            rec.update(labels=labels, error=err, status="ok" if labels else "failed")
        except FatalAPIError:
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_log(rec)
    stats[rec["status"]] += 1


async def classify_all(todo: pd.DataFrame):
    from tqdm import tqdm

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is empty in config.py")
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "X-Title":       "PluralValueBench",
    }
    sem = asyncio.Semaphore(config.CLASSIFIER_WORKERS)
    stats = Counter()
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(90.0)) as client:
        tasks = [asyncio.create_task(classify_one(client, sem, r, stats)) for r in todo.itertuples()]
        with tqdm(total=len(tasks), desc="  Classifying") as bar:
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
# ASSEMBLY
# =============================================================================

def assemble(questions: pd.DataFrame, allow_failures: bool):
    log = load_log()
    failed = [r.question_id for r in questions.itertuples() if not is_done(log.get(r.question_id, {}), r)]
    if failed and not allow_failures:
        raise SystemExit(
            f"\n  {len(failed)} questions have no valid label set (e.g. {failed[:5]}).\n"
            "  Re-run to retry them, inspect outputs/classifier_raw.jsonl, "
            "or pass --allow_failures."
        )

    out = []
    for r in questions.itertuples():
        rec = log.get(r.question_id, {})
        judg = rec.get("labels") if is_done(rec, r) else None
        row = {"question_id": r.question_id, "classification_failed": judg is None}
        for v in VALUES:
            row[f"is_{v}"] = judg[v] if judg else None
        labels = labels_list(judg) if judg else []
        row["schwartz_labels"] = json.dumps(labels)
        row["n_labels"] = len(labels)
        row["is_none"] = judg is not None and len(labels) == 0
        out.append(row)
    qdf = questions.merge(pd.DataFrame(out), on="question_id", how="left", validate="one_to_one")

    rows = pd.read_csv(config.Q0_ROWS_CSV)
    rows = rows[rows.question_id.isin(qdf.question_id)]      # subset when --limit is used
    n_expected = len(rows)
    label_cols = ["question_id", "schwartz_labels", "n_labels", "is_none", "classification_failed"]
    rows = rows.merge(qdf[label_cols], on="question_id", how="left", validate="many_to_one")
    assert len(rows) == n_expected, "row count changed during merge"
    assert rows.classification_failed.notna().all(), "rows without a classification record"

    ok = qdf[~qdf.classification_failed]
    q_counts = Counter(v for s in ok.schwartz_labels for v in json.loads(s))
    r_counts = Counter(v for s in rows.loc[~rows.classification_failed, "schwartz_labels"] for v in json.loads(s))
    recs = [log[q] for q in questions.question_id if q in log]
    tokens_in = sum(u.get("prompt_tokens", 0) for r in recs for u in r.get("usage", []))
    tokens_out = sum(u.get("completion_tokens", 0) for r in recs for u in r.get("usage", []))
    cost = sum(u.get("cost", 0) or 0 for r in recs for u in r.get("usage", []))
    report = {
        "classifier":            config.CLASSIFIER_MODEL_ID,
        "provider_config":       config.CLASSIFIER_PROVIDER,
        "providers_used":        dict(Counter(r.get("provider") for r in recs)),
        "protocol":              "binary per-value yes/no, multi-label (Kiesel et al., 2022)",
        "n_questions":           int(len(qdf)),
        "n_classified_ok":       int(len(ok)),
        "n_failed":              int(qdf.classification_failed.sum()),
        "n_needed_repair":       int(sum(1 for q in ok.question_id if log[q].get("repair_output") is not None)),
        "n_none":                int(ok.is_none.sum()),
        "share_multilabel":      round(float((ok.n_labels > 1).mean()), 4) if len(ok) else None,
        "mean_labels_per_question": round(float(ok.n_labels.mean()), 3) if len(ok) else None,
        "questions_per_value":   {v: int(q_counts.get(v, 0)) for v in VALUES},
        "rows_per_value":        {v: int(r_counts.get(v, 0)) for v in VALUES},
        "tokens_in":             int(tokens_in),
        "tokens_out":            int(tokens_out),
        "cost_usd_reported":     round(float(cost), 4),
    }
    return qdf, rows, report


def self_test():
    good = {v: False for v in VALUES}
    good.update({"Power": True, "Security": True})
    s, e = parse_labels("```json\n" + json.dumps(good) + "\n```")
    assert s is not None and labels_list(s) == ["Power", "Security"], e
    bad = dict(good); bad.pop("Power")
    assert parse_labels(json.dumps(bad))[0] is None                     # missing key
    bad = dict(good); bad["Power"] = 0.7
    assert parse_labels(json.dumps(bad))[0] is None                     # probability, not boolean
    bad = dict(good); bad["Power"] = 1
    assert parse_labels(json.dumps(bad))[0] is None                     # int is not a boolean
    yes = dict(good); yes["Power"] = "yes"
    assert parse_labels(json.dumps(yes))[0]["Power"] is True
    assert parse_labels("no json here")[0] is None
    print("  Parser self-test passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="classify only the first N questions (pilot)")
    ap.add_argument("--dry_run", action="store_true", help="print one prompt + run parser tests; no API calls")
    ap.add_argument("--allow_failures", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("PluralValueBench v2 — Stage I, Step 2: Schwartz classification")
    print("=" * 70)
    self_test()

    questions = pd.read_csv(config.Q0_QUESTIONS_CSV)
    if args.limit:
        questions = questions.head(args.limit)

    if args.dry_run:
        r = questions.iloc[0]
        print("\n--- System ---\n" + SYSTEM_PROMPT)
        print("\n--- Example prompt ---\n" + build_prompt(r.question_text, json.loads(r.options)))
        return

    log = load_log()
    todo = questions[[not is_done(log.get(r.question_id, {}), r) for r in questions.itertuples()]]
    print(f"  Model:     {config.CLASSIFIER_MODEL_ID}  (provider: {config.CLASSIFIER_PROVIDER['order']})")
    print(f"  Workers:   {config.CLASSIFIER_WORKERS}")
    print(f"  Questions: {len(questions):,}  |  done: {len(questions) - len(todo):,}  |  to do: {len(todo):,}")
    if len(todo):
        stats = asyncio.run(classify_all(todo))
        print(f"  This run: ok={stats['ok']}  failed={stats['failed']}")

    qdf, rows, report = assemble(questions, args.allow_failures)
    suffix = f"_pilot{args.limit}" if args.limit else ""
    q_path = config.Q1_QUESTIONS_CSV.with_name(config.Q1_QUESTIONS_CSV.stem + suffix + ".csv")
    r_path = config.Q1_ROWS_CSV.with_name(config.Q1_ROWS_CSV.stem + suffix + ".csv")
    rep_path = config.Q1_REPORT_JSON.with_name(config.Q1_REPORT_JSON.stem + suffix + ".json")
    qdf.to_csv(q_path, index=False)
    rows.to_csv(r_path, index=False)
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Classified OK: {report['n_classified_ok']:,} / {report['n_questions']:,}"
          f"  (repaired: {report['n_needed_repair']}, failed: {report['n_failed']})")
    print(f"  None (no value): {report['n_none']}   multi-label share: {report['share_multilabel']}")
    print(f"  Providers: {report['providers_used']}   cost reported: ${report['cost_usd_reported']}")
    print("  Questions per value:")
    for v, n in report["questions_per_value"].items():
        print(f"    {v:<15} {n:>5}")
    print(f"\n  Saved -> {q_path}\n  Saved -> {r_path}\n  Saved -> {rep_path}")


if __name__ == "__main__":
    main()
