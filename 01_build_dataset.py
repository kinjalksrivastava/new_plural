"""
===============================================================================
PluralValueBench v2 — Stage I, Step 1: Build the cleaned corpus (Q0)
===============================================================================

Downloads GlobalOpinionQA (Durmus et al., 2023) at a pinned Hugging Face
revision and produces one row per (question, country) with the real human
answer distribution.

Cleaning rules (every one is counted in Q0_report.json):
  1. Non-national samples ("X (Non-national sample)") are dropped.
  2. Country-name variants are merged to one canonical name
     (Britain / Great Britain, Czech Rep. / Czechia, S. Korea / South Korea,
     Taiwan / Taiwan ROC, S. Africa -> South Africa, ...).
  3. India has two national samples. "Current national sample" is kept;
     "Old national sample" is used only when the current one is absent.
  4. All-zero distributions (no data) are dropped.
  5. Distributions are renormalised to sum to 1; rows whose raw sum was off by
     more than DIST_SUM_TOLERANCE are flagged (dist_renormalized = True).
  6. Numeric option labels stored as floats (0.0, 1.0, ...) are rendered as
     "0", "1", ... so prompts never show "1.0".
  7. Non-substantive options (DK / Refused / No answer / Missing) are KEPT
     (GlobalOpinionQA convention) but their indices are recorded in
     nonsubstantive_idx so later stages can analyse them explicitly.
  8. Questions left with zero countries after cleaning are dropped.

Every unknown country label raises an error (no silent drops).

Outputs:
  data/raw/global_opinions.csv   cached raw download
  data/Q0_rows.csv               one row per (question, country)
  data/Q0_questions.csv          one row per question (classifier input)
  data/Q0_report.json            all counts + SHA-256 of the raw file

Usage:
  python 01_build_dataset.py
  python 01_build_dataset.py --force_download
===============================================================================
"""

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

import config

# =============================================================================
# COUNTRY CANONICALISATION
# raw label -> (canonical name, ISO-3166 alpha-3 or None, unit type)
# unit type: "country" | "territory" (sub-national or non-sovereign unit)
# =============================================================================
COUNTRY_MAP = {
    "Albania":                         ("Albania",                 "ALB", "country"),
    "Andorra":                         ("Andorra",                 "AND", "country"),
    "Angola":                          ("Angola",                  "AGO", "country"),
    "Argentina":                       ("Argentina",               "ARG", "country"),
    "Armenia":                         ("Armenia",                 "ARM", "country"),
    "Australia":                       ("Australia",               "AUS", "country"),
    "Austria":                         ("Austria",                 "AUT", "country"),
    "Azerbaijan":                      ("Azerbaijan",              "AZE", "country"),
    "Bangladesh":                      ("Bangladesh",              "BGD", "country"),
    "Belarus":                         ("Belarus",                 "BLR", "country"),
    "Belgium":                         ("Belgium",                 "BEL", "country"),
    "Bolivia":                         ("Bolivia",                 "BOL", "country"),
    "Bosnia Herzegovina":              ("Bosnia and Herzegovina",  "BIH", "country"),
    "Brazil":                          ("Brazil",                  "BRA", "country"),
    "Britain":                         ("Great Britain",           "GBR", "country"),
    "Great Britain":                   ("Great Britain",           "GBR", "country"),
    "Bulgaria":                        ("Bulgaria",                "BGR", "country"),
    "Burkina Faso":                    ("Burkina Faso",            "BFA", "country"),
    "Canada":                          ("Canada",                  "CAN", "country"),
    "Chile":                           ("Chile",                   "CHL", "country"),
    "China":                           ("China",                   "CHN", "country"),
    "Colombia":                        ("Colombia",                "COL", "country"),
    "Croatia":                         ("Croatia",                 "HRV", "country"),
    "Cyprus":                          ("Cyprus",                  "CYP", "country"),
    "Czech Rep.":                      ("Czech Republic",          "CZE", "country"),
    "Czechia":                         ("Czech Republic",          "CZE", "country"),
    "Denmark":                         ("Denmark",                 "DNK", "country"),
    "Ecuador":                         ("Ecuador",                 "ECU", "country"),
    "Egypt":                           ("Egypt",                   "EGY", "country"),
    "El Salvador":                     ("El Salvador",             "SLV", "country"),
    "Estonia":                         ("Estonia",                 "EST", "country"),
    "Ethiopia":                        ("Ethiopia",                "ETH", "country"),
    "Finland":                         ("Finland",                 "FIN", "country"),
    "France":                          ("France",                  "FRA", "country"),
    "Georgia":                         ("Georgia",                 "GEO", "country"),
    "Germany":                         ("Germany",                 "DEU", "country"),
    "Ghana":                           ("Ghana",                   "GHA", "country"),
    "Greece":                          ("Greece",                  "GRC", "country"),
    "Guatemala":                       ("Guatemala",               "GTM", "country"),
    "Honduras":                        ("Honduras",                "HND", "country"),
    "Hong Kong SAR":                   ("Hong Kong",               "HKG", "territory"),
    "Hungary":                         ("Hungary",                 "HUN", "country"),
    "Iceland":                         ("Iceland",                 "ISL", "country"),
    "India (Current national sample)": ("India",                   "IND", "country"),
    "India (Old national sample)":     ("India",                   "IND", "country"),
    "Indonesia":                       ("Indonesia",               "IDN", "country"),
    "Iran":                            ("Iran",                    "IRN", "country"),
    "Iraq":                            ("Iraq",                    "IRQ", "country"),
    "Israel":                          ("Israel",                  "ISR", "country"),
    "Italy":                           ("Italy",                   "ITA", "country"),
    "Ivory Coast":                     ("Ivory Coast",             "CIV", "country"),
    "Japan":                           ("Japan",                   "JPN", "country"),
    "Jordan":                          ("Jordan",                  "JOR", "country"),
    "Kazakhstan":                      ("Kazakhstan",              "KAZ", "country"),
    "Kenya":                           ("Kenya",                   "KEN", "country"),
    "Kuwait":                          ("Kuwait",                  "KWT", "country"),
    "Kyrgyzstan":                      ("Kyrgyzstan",              "KGZ", "country"),
    "Latvia":                          ("Latvia",                  "LVA", "country"),
    "Lebanon":                         ("Lebanon",                 "LBN", "country"),
    "Libya":                           ("Libya",                   "LBY", "country"),
    "Lithuania":                       ("Lithuania",               "LTU", "country"),
    "Macau SAR":                       ("Macau",                   "MAC", "territory"),
    "Malaysia":                        ("Malaysia",                "MYS", "country"),
    "Maldives":                        ("Maldives",                "MDV", "country"),
    "Mali":                            ("Mali",                    "MLI", "country"),
    "Mexico":                          ("Mexico",                  "MEX", "country"),
    "Mongolia":                        ("Mongolia",                "MNG", "country"),
    "Montenegro":                      ("Montenegro",              "MNE", "country"),
    "Morocco":                         ("Morocco",                 "MAR", "country"),
    "Myanmar":                         ("Myanmar",                 "MMR", "country"),
    "Netherlands":                     ("Netherlands",             "NLD", "country"),
    "New Zealand":                     ("New Zealand",             "NZL", "country"),
    "Nicaragua":                       ("Nicaragua",               "NIC", "country"),
    "Nigeria":                         ("Nigeria",                 "NGA", "country"),
    "North Macedonia":                 ("North Macedonia",         "MKD", "country"),
    "Northern Ireland":                ("Northern Ireland",        None,  "territory"),
    "Norway":                          ("Norway",                  "NOR", "country"),
    "Pakistan":                        ("Pakistan",                "PAK", "country"),
    "Palest. ter.":                    ("Palestinian Territories", "PSE", "territory"),
    "Peru":                            ("Peru",                    "PER", "country"),
    "Philippines":                     ("Philippines",             "PHL", "country"),
    "Poland":                          ("Poland",                  "POL", "country"),
    "Portugal":                        ("Portugal",                "PRT", "country"),
    "Puerto Rico":                     ("Puerto Rico",             "PRI", "territory"),
    "Romania":                         ("Romania",                 "ROU", "country"),
    "Russia":                          ("Russia",                  "RUS", "country"),
    "S. Africa":                       ("South Africa",            "ZAF", "country"),
    "S. Korea":                        ("South Korea",             "KOR", "country"),
    "South Korea":                     ("South Korea",             "KOR", "country"),
    "Senegal":                         ("Senegal",                 "SEN", "country"),
    "Serbia":                          ("Serbia",                  "SRB", "country"),
    "Singapore":                       ("Singapore",               "SGP", "country"),
    "Slovakia":                        ("Slovakia",                "SVK", "country"),
    "Slovenia":                        ("Slovenia",                "SVN", "country"),
    "Spain":                           ("Spain",                   "ESP", "country"),
    "Sweden":                          ("Sweden",                  "SWE", "country"),
    "Switzerland":                     ("Switzerland",             "CHE", "country"),
    "Taiwan":                          ("Taiwan",                  "TWN", "country"),
    "Taiwan ROC":                      ("Taiwan",                  "TWN", "country"),
    "Tajikistan":                      ("Tajikistan",              "TJK", "country"),
    "Tanzania":                        ("Tanzania",                "TZA", "country"),
    "Thailand":                        ("Thailand",                "THA", "country"),
    "Tunisia":                         ("Tunisia",                 "TUN", "country"),
    "Turkey":                          ("Turkey",                  "TUR", "country"),
    "Uganda":                          ("Uganda",                  "UGA", "country"),
    "Ukraine":                         ("Ukraine",                 "UKR", "country"),
    "United States":                   ("United States",           "USA", "country"),
    "Uruguay":                         ("Uruguay",                 "URY", "country"),
    "Uzbekistan":                      ("Uzbekistan",              "UZB", "country"),
    "Venezuela":                       ("Venezuela",               "VEN", "country"),
    "Vietnam":                         ("Vietnam",                 "VNM", "country"),
    "Zimbabwe":                        ("Zimbabwe",                "ZWE", "country"),
}

INDIA_CURRENT = "India (Current national sample)"
INDIA_OLD     = "India (Old national sample)"
NON_NATIONAL_PATTERN = re.compile(r"\(non-national sample\)", re.IGNORECASE)

# Non-substantive response options (exact labels found in GlobalOpinionQA).
# Content options that merely contain "know" (e.g. "Don't know anyone (VOL)")
# are deliberately NOT in this list.
NONSUBSTANTIVE_OPTIONS = {
    "DK/Refused", "Don't know/Refused", "Don't know", "Don't Know",
    "I don't know", "Don't know/ Unsure", "Don't know/No answer",
    "No answer", "No Answer", "No answer/refused", "No answer/Refused",
    "No answer/ refused", "Missing", "Missing; Not available",
    "Missing; Unknown", "Missing; Not applicable for other reasons",
    "Missing; Not available; Not applicable",
    "Other missing; Multiple answers Mail (EVS)",
    "Other missing; Not applicable; Multiple answers Mail (EVS)",
}


# =============================================================================
# PARSING HELPERS
# =============================================================================

def parse_selections(raw: str) -> dict:
    """Parse "defaultdict(<class 'list'>, {...})" into a plain dict."""
    s = raw.strip()
    prefix = "defaultdict(<class 'list'>,"
    if s.startswith(prefix):
        s = s[len(prefix):].strip()
        if not s.endswith(")"):
            raise ValueError(f"Malformed selections string: {raw[:80]}")
        s = s[:-1]
    out = ast.literal_eval(s)
    if not isinstance(out, dict):
        raise ValueError(f"Selections did not parse to a dict: {raw[:80]}")
    return out


def render_option(opt) -> str:
    """Render an option label as a clean string (0.0 -> '0')."""
    if isinstance(opt, float):
        if not np.isfinite(opt):
            raise ValueError(f"Non-finite option label: {opt}")
        return str(int(opt)) if opt.is_integer() else str(opt)
    if isinstance(opt, int):
        return str(opt)
    if not isinstance(opt, str):
        raise ValueError(f"Unexpected option type {type(opt)}: {opt!r}")
    return opt.strip()


def parse_options(raw: str) -> list:
    opts = ast.literal_eval(raw)
    if not isinstance(opts, list) or len(opts) < 2:
        raise ValueError(f"Options must be a list with >= 2 entries: {raw[:80]}")
    opts = [render_option(o) for o in opts]
    if any(o == "" for o in opts):
        raise ValueError(f"Empty option label: {raw[:80]}")
    return opts


def merged_variant_questions(df: pd.DataFrame) -> list:
    """
    Questions whose option list merges different survey variants under one ID
    (e.g. Q1705: Islam/law, EU membership and human-rights items; Q0662: "Good"
    vs "Good thing"). Signature: two countries of the question put mass on
    DISJOINT sets of substantive options. Such questions have no common option
    space, so they are removed.
    """
    from itertools import combinations
    out = []
    for qid, g in df.groupby("question_id", sort=False):
        if len(g) < 2:
            continue
        opts = json.loads(g.options.iloc[0])
        ns = {i for i, o in enumerate(opts) if o in NONSUBSTANTIVE_OPTIONS}
        sup = [frozenset(i for i, x in enumerate(json.loads(h)) if x > 0 and i not in ns)
               for h in g.human_distribution]
        if any(a and b and not (a & b) for a, b in combinations(sup, 2)):
            out.append(qid)
    return out


TOTAL_ROW_PATTERN = re.compile(r"^\s*total\b.*N=\s*$", re.IGNORECASE)


def prune_options(df: pd.DataFrame):
    """
    Remove answer options that are not real response categories, per question:
      (a) table-total rows carried over from Pew reports ("Total   N="), whose
          "share" is the whole sample; their removal restores the real distribution;
      (b) options chosen by no respondent in ANY country of the question. These
          carry no human signal and include options concatenated from other
          questions in the source (e.g. Q1705, Q0256).
    Remaining shares are renormalised. Questions left with < 2 options are dropped.
    """
    info = {"total_row_questions": [], "never_chosen_option_questions": [],
            "dropped_questions_lt2_options": []}
    out = []
    for qid, g in df.groupby("question_id", sort=False):
        opts = json.loads(g.options.iloc[0])
        P = np.stack(g.human_distribution.apply(lambda s: np.array(json.loads(s))).values)
        is_total = [bool(TOTAL_ROW_PATTERN.match(o)) for o in opts]
        never = list((P == 0).all(axis=0))
        keep = [i for i in range(len(opts)) if not is_total[i] and not never[i]]
        if any(is_total):
            info["total_row_questions"].append(qid)
        if any(n and not t for n, t in zip(never, is_total)):
            info["never_chosen_option_questions"].append(qid)
        if len(keep) == len(opts):
            out.append(g)
            continue
        if len(keep) < 2:
            info["dropped_questions_lt2_options"].append(qid)
            continue
        new_opts = [opts[i] for i in keep]
        Pk = P[:, keep]
        Pk = Pk / Pk.sum(axis=1, keepdims=True)
        g = g.copy()
        g["options"] = json.dumps(new_opts, ensure_ascii=False)
        g["n_options"] = len(new_opts)
        g["nonsubstantive_idx"] = json.dumps([j for j, o in enumerate(new_opts) if o in NONSUBSTANTIVE_OPTIONS])
        g["human_distribution"] = [json.dumps([round(float(x), 6) for x in row]) for row in Pk]
        out.append(g)
    info["n_total_row_questions"] = len(info["total_row_questions"])
    info["n_never_chosen_option_questions"] = len(info["never_chosen_option_questions"])
    return pd.concat(out, ignore_index=True), info


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# MAIN BUILD
# =============================================================================

def download(force: bool):
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_GOQA_CSV.exists() and not force:
        print(f"  Using cached raw file: {config.RAW_GOQA_CSV}")
        return
    print(f"  Downloading GlobalOpinionQA @ {config.GOQA_REVISION[:10]} ...")
    req = urllib.request.Request(config.GOQA_URL)
    if config.HF_TOKEN:
        req.add_header("Authorization", f"Bearer {config.HF_TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as resp, open(config.RAW_GOQA_CSV, "wb") as f:
        f.write(resp.read())
    print(f"  Saved -> {config.RAW_GOQA_CSV}")


def build():
    # keep_default_na=False: empty cells stay "" (never NaN -> "nan")
    raw = pd.read_csv(config.RAW_GOQA_CSV, keep_default_na=False)
    expected_cols = {"question", "selections", "options", "source"}
    if set(raw.columns) != expected_cols:
        raise ValueError(f"Unexpected columns {list(raw.columns)}")

    c = Counter()                      # cleaning counters
    unknown_labels = set()
    empty_text_qids = []
    rows = []

    for src_idx, rec in raw.iterrows():
        qid = f"Q{src_idx:04d}"        # same ID scheme as v1 (source row index)
        qtext = str(rec["question"]).strip()
        if not qtext:
            empty_text_qids.append(qid)
            continue
        source = str(rec["source"]).strip()
        options = parse_options(rec["options"])
        k = len(options)
        nonsub_idx = [i for i, o in enumerate(options) if o in NONSUBSTANTIVE_OPTIONS]
        selections = parse_selections(rec["selections"])
        c["raw_question_country_pairs"] += len(selections)

        # India: prefer the current national sample when both exist
        if INDIA_CURRENT in selections and INDIA_OLD in selections:
            selections = {k_: v for k_, v in selections.items() if k_ != INDIA_OLD}
            c["dropped_india_old_when_current_present"] += 1

        seen_canonical = {}
        for label, dist in selections.items():
            if NON_NATIONAL_PATTERN.search(label):
                c["dropped_non_national"] += 1
                continue
            if label not in COUNTRY_MAP:
                unknown_labels.add(label)
                continue
            canon, iso3, unit_type = COUNTRY_MAP[label]

            if canon in seen_canonical:
                raise ValueError(
                    f"{qid}: two raw labels map to '{canon}': "
                    f"{seen_canonical[canon]!r} and {label!r}"
                )
            seen_canonical[canon] = label

            p = np.asarray(dist, dtype=float)
            if p.shape != (k,):
                raise ValueError(f"{qid}/{label}: distribution length {p.size} != {k} options")
            if not np.all(np.isfinite(p)) or np.any(p < 0):
                raise ValueError(f"{qid}/{label}: invalid values in distribution {dist}")
            total = float(p.sum())
            if total == 0.0:
                c["dropped_all_zero_distribution"] += 1
                continue
            renorm = abs(total - 1.0) > config.DIST_SUM_TOLERANCE
            c["flagged_renormalized"] += int(renorm)
            p = p / total

            rows.append({
                "question_id":        qid,
                "source":             source,
                "question_text":      qtext,
                "options":            json.dumps(options, ensure_ascii=False),
                "n_options":          k,
                "nonsubstantive_idx": json.dumps(nonsub_idx),
                "country":            canon,
                "iso3":               iso3,
                "unit_type":          unit_type,
                "raw_country_label":  label,
                "human_distribution": json.dumps([round(float(x), 6) for x in p]),
                "raw_dist_sum":       round(total, 6),
                "dist_renormalized":  renorm,
            })

    if unknown_labels:
        raise ValueError(f"Unmapped country labels (add them to COUNTRY_MAP): {sorted(unknown_labels)}")

    df = pd.DataFrame(rows)
    df, prune_info = prune_options(df)
    merged = merged_variant_questions(df)
    prune_info["merged_variant_questions_removed"] = merged
    prune_info["merged_variant_rows_removed"] = int(df.question_id.isin(merged).sum())
    df = df[~df.question_id.isin(merged)].reset_index(drop=True)
    n_countries = df.groupby("question_id")["country"].nunique()
    df["n_countries_for_question"] = df["question_id"].map(n_countries)

    validate(df)

    questions = (
        df.drop_duplicates("question_id")[
            ["question_id", "source", "question_text", "options", "n_options",
             "nonsubstantive_idx", "n_countries_for_question"]
        ]
        .rename(columns={"n_countries_for_question": "n_countries"})
        .reset_index(drop=True)
    )
    dropped_questions = sorted(set(f"Q{i:04d}" for i in range(len(raw)))
                               - set(questions["question_id"]) - set(empty_text_qids)
                               - set(prune_info["merged_variant_questions_removed"]))

    report = {
        "source_url":                 config.GOQA_URL,
        "raw_sha256":                 sha256_of(config.RAW_GOQA_CSV),
        "raw_questions":              int(len(raw)),
        "raw_question_country_pairs": int(c["raw_question_country_pairs"]),
        "dropped_non_national":       int(c["dropped_non_national"]),
        "dropped_india_old_when_current_present": int(c["dropped_india_old_when_current_present"]),
        "dropped_all_zero_distribution": int(c["dropped_all_zero_distribution"]),
        "option_pruning":             prune_info,
        "dropped_empty_question_text": empty_text_qids,
        "flagged_renormalized":       int(c["flagged_renormalized"]),
        "dropped_questions_no_country_left": dropped_questions,
        "final_rows":                 int(len(df)),
        "final_questions":            int(questions["question_id"].nunique()),
        "final_units":                int(df["country"].nunique()),
        "final_countries":            int(df.loc[df.unit_type == "country", "country"].nunique()),
        "final_territories":          sorted(df.loc[df.unit_type == "territory", "country"].unique().tolist()),
        "rows_by_source":             df["source"].value_counts().to_dict(),
        "questions_by_source":        questions["source"].value_counts().to_dict(),
        "questions_single_country":   int((questions["n_countries"] == 1).sum()),
        "questions_with_nonsubstantive_option": int((questions["nonsubstantive_idx"] != "[]").sum()),
        "n_options_histogram":        {int(k): int(v) for k, v in questions["n_options"].value_counts().sort_index().items()},
        "rows_per_unit":              df["country"].value_counts().sort_index().to_dict(),
    }
    return df, questions, report


def validate(df: pd.DataFrame):
    """Hard checks. Any failure stops the pipeline."""
    assert not df.duplicated(["question_id", "country"]).any(), "duplicate (question, country) rows"
    for _, r in df.iterrows():
        p = np.array(json.loads(r["human_distribution"]))
        opts = json.loads(r["options"])
        assert len(p) == len(opts) == r["n_options"], f"length mismatch at {r.question_id}/{r.country}"
        assert abs(p.sum() - 1.0) < 1e-4, f"distribution does not sum to 1 at {r.question_id}/{r.country}"
        assert (p >= 0).all(), f"negative mass at {r.question_id}/{r.country}"
    assert df["question_text"].notna().all() and (df["question_text"].str.strip() != "").all(), "empty question text"
    assert not df["question_text"].str.lower().isin(["nan", "none", "null"]).any(), "placeholder question text"
    # each question has one text/options combination
    assert (df.groupby("question_id")[["question_text", "options"]].nunique() == 1).all().all()
    print(f"  Validation passed on {len(df):,} rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_download", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("PluralValueBench v2 — Stage I, Step 1: build Q0")
    print("=" * 70)
    download(args.force_download)
    df, questions, report = build()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.Q0_ROWS_CSV, index=False)
    questions.to_csv(config.Q0_QUESTIONS_CSV, index=False)
    with open(config.Q0_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Raw questions:                {report['raw_questions']:>7,}")
    print(f"  Raw (question, country) pairs:{report['raw_question_country_pairs']:>7,}")
    print(f"  Dropped non-national:         {report['dropped_non_national']:>7,}")
    print(f"  Dropped India old (dup):      {report['dropped_india_old_when_current_present']:>7,}")
    print(f"  Dropped all-zero dists:       {report['dropped_all_zero_distribution']:>7,}")
    print(f"  Renormalised (flagged):       {report['flagged_renormalized']:>7,}")
    print(f"  Questions with 'Total N=' option removed: {report['option_pruning']['n_total_row_questions']}")
    print(f"  Questions with never-chosen options removed: {report['option_pruning']['n_never_chosen_option_questions']}")
    print(f"  Merged-variant questions removed: {len(report['option_pruning']['merged_variant_questions_removed'])} ({report['option_pruning']['merged_variant_rows_removed']} rows)")
    print(f"  Questions dropped (no text):  {len(report['dropped_empty_question_text']):>7,}")
    print(f"  Questions dropped (no country):{len(report['dropped_questions_no_country_left']):>6,}")
    print(f"  Final rows:                   {report['final_rows']:>7,}")
    print(f"  Final questions:              {report['final_questions']:>7,}")
    print(f"  Final units:                  {report['final_units']:>7,}  "
          f"({report['final_countries']} countries + {len(report['final_territories'])} territories)")
    print(f"\n  Saved -> {config.Q0_ROWS_CSV}")
    print(f"  Saved -> {config.Q0_QUESTIONS_CSV}")
    print(f"  Saved -> {config.Q0_REPORT_JSON}")


if __name__ == "__main__":
    main()
