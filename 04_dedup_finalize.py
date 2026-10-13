"""
===============================================================================
PluralValueBench v2 — Stage I, Step 4: Deduplication and final corpus (Q3)
===============================================================================

Near-duplicate detection with 64-bit SimHash (Sadowski & Levin, 2007) over
word 3-gram shingles of question text + options, as in Demo-SafetyBench
(tau = 10 Hamming bits).

GlobalOpinionQA contains many survey *batteries*: a long shared stem followed
by a short item ("...Air pollution" vs "...Health care"). SimHash on full text
flags such siblings as near-duplicates although they are distinct questions
with distinct human distributions. Therefore SimHash only proposes CANDIDATE
pairs; a candidate is a TRUE duplicate only if
  (a) the normalised question texts are identical (case, punctuation,
      whitespace, brackets ignored), AND
  (b) the answer options are identical.
For each true-duplicate pair the question with fewer countries is removed
(ties: the later question_id). A sweep over SIMHASH_TAU_SWEEP is reported
(candidates vs verified duplicates), mirroring Demo-SafetyBench's tau analysis.

Outputs:
  data/Q3_final_rows.csv           FINAL Stage I corpus (one row per question×country)
  data/Q3_final_questions.csv      one row per question
  data/Q3_simhash_candidates.csv   all candidate pairs at the max tau, with verdicts
  data/Q3_report.json              dedup sweep + final corpus statistics
  data/Q3_table_stage1.tex         Stage I domain-progression table (Q1 -> Q2 -> Q3)

Usage:
  python 04_dedup_finalize.py
===============================================================================
"""

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

import config

VALUES = config.SCHWARTZ_VALUES


# =============================================================================
# SIMHASH
# =============================================================================

def tokens(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def normalise(text: str) -> str:
    return " ".join(tokens(text))


def simhash(text: str) -> int:
    toks = tokens(text)
    n = config.SIMHASH_SHINGLE
    shingles = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)] or [" ".join(toks)]
    v = np.zeros(config.SIMHASH_BITS)
    for s in shingles:
        h = int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big")
        bits = np.array([(h >> b) & 1 for b in range(config.SIMHASH_BITS)])
        v += np.where(bits == 1, 1, -1)
    return int(sum(1 << b for b in range(config.SIMHASH_BITS) if v[b] > 0))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# =============================================================================
# DEDUP
# =============================================================================

def find_candidates(q: pd.DataFrame, max_tau: int) -> pd.DataFrame:
    hashes = [simhash(t + " || " + " | ".join(json.loads(o)))
              for t, o in zip(q.question_text, q.options)]
    norm = [normalise(t) for t in q.question_text]
    rows = []
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            d = hamming(hashes[i], hashes[j])
            if d <= max_tau:
                same_text = norm[i] == norm[j]
                same_opts = q.options.iloc[i] == q.options.iloc[j]
                rows.append({
                    "qid_a": q.question_id.iloc[i], "qid_b": q.question_id.iloc[j],
                    "hamming": d, "same_text": same_text, "same_options": same_opts,
                    "true_duplicate": same_text and same_opts,
                    "text_a_tail": q.question_text.iloc[i][-80:],
                    "text_b_tail": q.question_text.iloc[j][-80:],
                })
    return pd.DataFrame(rows, columns=["qid_a", "qid_b", "hamming", "same_text", "same_options",
                                       "true_duplicate", "text_a_tail", "text_b_tail"])


def choose_removals(dups: pd.DataFrame, n_countries: dict) -> set:
    remove = set()
    for r in dups.itertuples():
        if r.qid_a in remove or r.qid_b in remove:
            continue
        a, b = r.qid_a, r.qid_b
        keep_a = (n_countries[a], b) >= (n_countries[b], a)   # more countries wins; tie -> earlier id
        remove.add(b if keep_a else a)
    return remove


# =============================================================================
# REPORTING
# =============================================================================

def per_value_counts(rows: pd.DataFrame) -> tuple:
    q = rows.drop_duplicates("question_id")
    qc = Counter(v for s in q.schwartz_labels for v in json.loads(s))
    rc = Counter(v for s in rows.schwartz_labels for v in json.loads(s))
    return ({v: int(qc.get(v, 0)) for v in VALUES}, {v: int(rc.get(v, 0)) for v in VALUES})


def latex_table(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> str:
    (_, r1), (_, r2), (q3q, r3) = per_value_counts(q1), per_value_counts(q2), per_value_counts(q3)
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Value & $Q_1$ rows & +Expansion & $Q_3$ rows & $Q_3$ questions \\", r"\midrule",
    ]
    for v in VALUES:
        add = r2[v] - r1[v]
        lines.append(f"{v} & {r1[v]:,} & {'+' + format(add, ',') if add else '0'} & {r3[v]:,} & {q3q[v]:,} \\\\")
    lines += [
        r"\midrule",
        f"None & {int(q1.is_none.sum()):,} & 0 & {int(q3.is_none.sum()):,} & "
        f"{int(q3.drop_duplicates('question_id').is_none.sum()):,} \\\\",
        f"Total (unique) & {len(q1):,} & +{len(q2) - len(q1):,} & {len(q3):,} & {q3.question_id.nunique():,} \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Stage I corpus progression. Row counts are (question, country) pairs; "
        r"a multi-label question is counted once per assigned value, so value rows sum to more than the total.}",
        r"\label{tab:stage1}", r"\end{table}",
    ]
    return "\n".join(lines)


def validate(rows: pd.DataFrame):
    assert not rows.duplicated(["question_id", "country"]).any(), "duplicate (question, country)"
    assert not rows.classification_failed.any(), "unclassified rows in final corpus"
    assert rows.question_text.notna().all() and (rows.question_text.str.strip() != "").all()
    for p, o in zip(rows.human_distribution, rows.options):
        p, o = np.array(json.loads(p)), json.loads(o)
        assert len(p) == len(o) and abs(p.sum() - 1) < 1e-4 and (p >= 0).all()
    print(f"  Validation passed on {len(rows):,} rows.")


def main():
    print("=" * 70)
    print("PluralValueBench v2 — Stage I, Step 4: dedup + final corpus")
    print("=" * 70)

    q1 = pd.read_csv(config.Q1_ROWS_CSV)
    q2 = pd.read_csv(config.Q2_ROWS_CSV)
    questions = q2.drop_duplicates("question_id").reset_index(drop=True)
    print(f"  Q2: {len(q2):,} rows, {len(questions):,} questions")

    max_tau = max(config.SIMHASH_TAU_SWEEP)
    print(f"  SimHash ({config.SIMHASH_BITS}-bit, word {config.SIMHASH_SHINGLE}-grams), "
          f"pairs with Hamming <= {max_tau} ...")
    cands = find_candidates(questions, max_tau)

    sweep = {}
    for tau in config.SIMHASH_TAU_SWEEP:
        c = cands[cands.hamming <= tau]
        sweep[tau] = {"candidate_pairs": int(len(c)),
                      "verified_duplicates": int(c.true_duplicate.sum()),
                      "battery_siblings_or_distinct": int((~c.true_duplicate).sum())}
        print(f"    tau={tau:>2}: candidates={len(c):>6,}  verified duplicates={int(c.true_duplicate.sum())}")

    n_countries = q2.groupby("question_id").country.nunique().to_dict()
    dups = cands[(cands.hamming <= config.SIMHASH_TAU) & cands.true_duplicate]
    removed = choose_removals(dups, n_countries)
    q3 = q2[~q2.question_id.isin(removed)].reset_index(drop=True)
    validate(q3)

    q3_questions = q3.drop_duplicates("question_id").drop(
        columns=["country", "iso3", "unit_type", "raw_country_label", "human_distribution",
                 "raw_dist_sum", "dist_renormalized", "n_respondents"])
    qv, rv = per_value_counts(q3)
    report = {
        "simhash": {"bits": config.SIMHASH_BITS, "shingle": config.SIMHASH_SHINGLE,
                    "tau_used": config.SIMHASH_TAU, "sweep": sweep},
        "verification_rule": "true duplicate iff normalised question text AND options identical",
        "removed_questions": sorted(removed),
        "final_rows": int(len(q3)),
        "final_questions": int(q3.question_id.nunique()),
        "final_units": int(q3.country.nunique()),
        "rows_by_source": q3.source.value_counts().to_dict(),
        "questions_by_source": q3.drop_duplicates("question_id").source.value_counts().to_dict(),
        "expansion_rows": int(q3.is_expansion.sum()),
        "none_questions": int(q3.drop_duplicates("question_id").is_none.sum()),
        "questions_per_value": qv,
        "rows_per_value": rv,
        "mean_labels_per_question": round(float(q3.drop_duplicates("question_id").n_labels.mean()), 3),
    }

    q3.to_csv(config.Q3_ROWS_CSV, index=False)
    q3_questions.to_csv(config.Q3_QUESTIONS_CSV, index=False)
    cands.to_csv(config.Q3_DEDUP_PAIRS_CSV, index=False)
    with open(config.Q3_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    with open(config.Q3_TABLE_TEX, "w") as f:
        f.write(latex_table(q1, q2, q3))

    print(f"\n  Removed duplicates: {len(removed)} {sorted(removed)}")
    print(f"  FINAL Q3: {report['final_rows']:,} rows | {report['final_questions']:,} questions | "
          f"{report['final_units']} countries/territories")
    print("  Questions / rows per value:")
    for v in VALUES:
        print(f"    {v:<15} {qv[v]:>5}  {rv[v]:>7,}")
    for p in [config.Q3_ROWS_CSV, config.Q3_QUESTIONS_CSV, config.Q3_DEDUP_PAIRS_CSV,
              config.Q3_REPORT_JSON, config.Q3_TABLE_TEX]:
        print(f"  Saved -> {p}")


if __name__ == "__main__":
    main()
