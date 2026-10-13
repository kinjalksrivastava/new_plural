"""
===============================================================================
PluralValueBench v2 — Stage II, Step 8: Metrics
===============================================================================

All metrics are computed on the rows that EVERY rater answered successfully
(common set), excluding structural-zero rows. h = human distribution,
m = model distribution, K = number of options.

Fidelity (Durmus et al., 2023):  similarity = 1 - JS distance(h, m),
    JS distance = sqrt(JSD_2), JSD with base-2 logarithms (bounded [0, 1]).
Baselines (Meister et al., 2025; Cao et al., 2025; Suh et al., 2025):
    uniform            1/K on every option
    country-agnostic   equal-weight mean of the OTHER countries' human
                       distributions for the same question (leave-one-out);
                       defined for questions asked in >= 2 countries.
Top-1 match: argmax(m) == argmax(h).
Dispersion (entropy gap): Hn(m) - Hn(h), Hn = Shannon entropy / log K (in [0, 1]).
    > 0: model more spread out than humans; < 0: more peaked.
Between-country divergence (Chae et al., 2026, CDP):
    centroid p̄_q = equal-weight mean of country distributions for question q
    CD = mean over (q, c) of JSD_2(p_{c,q} || p̄_q)
    CDP = CD_model / CD_human * 100   (<100 flattening, >100 caricature)
    computed on questions with >= 2 countries; per region using only that
    region's countries (questions with >= 2 of them).
Inter-rater: mean JS distance between two models' distributions on the same row.
Schwartz orientations: Spearman correlation, across countries with scores,
    between a country's mean similarity and each of the 7 orientation scores.
    Also for similarity minus the country-agnostic baseline (adjusted), which
    removes how close a country is to the cross-country mean.
Uncertainty: 95% cluster-bootstrap CIs over questions (config.BOOTSTRAP_B);
    paired bootstrap for model differences, Holm-corrected p-values.

Usage:
  python 08_metrics.py --pilot
  python 08_metrics.py
===============================================================================
"""

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import stage2_common as sc

MODELS = list(config.OPENAI_RATERS) + list(config.OPENROUTER_RATERS)
ORIENT_COLS = ["or_harmony", "or_embeddedness", "or_hierarchy", "or_mastery",
               "or_affective_autonomy", "or_intellectual_autonomy", "or_egalitarianism"]


# =============================================================================
# DISTRIBUTION FUNCTIONS
# =============================================================================

def jsd2(p, q):
    """Jensen-Shannon divergence, base 2, in [0, 1]. Zeros handled exactly."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def js_distance(p, q):
    return float(np.sqrt(jsd2(p, q)))


def norm_entropy(p):
    k = len(p)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)) / np.log(k))


def unit_tests():
    p = np.array([0.5, 0.5]); q = np.array([1.0, 0.0])
    assert abs(jsd2(p, p)) < 1e-12
    assert abs(jsd2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-12   # max = 1 in base 2
    assert abs(jsd2(p, q) - jsd2(q, p)) < 1e-12                                     # symmetric
    assert abs(norm_entropy(np.array([0.25] * 4)) - 1.0) < 1e-12
    assert norm_entropy(np.array([1.0, 0.0, 0.0])) == 0.0
    from scipy.spatial.distance import jensenshannon
    r = np.random.default_rng(0)
    for _ in range(50):
        a, b = r.dirichlet(np.ones(5)), r.dirichlet(np.ones(5))
        assert abs(js_distance(a, b) - jensenshannon(a, b, base=2)) < 1e-9          # matches scipy
    print("  Metric unit tests passed (incl. agreement with scipy jensenshannon, base 2).")


# =============================================================================
# LOAD
# =============================================================================

def load(pilot: bool) -> pd.DataFrame:
    rows = sc.select_rows(pilot, None)
    rows["h"] = rows.human_distribution.apply(lambda s: np.array(json.loads(s)))
    rows["labels"] = rows.schwartz_labels.apply(json.loads)
    coverage = {}
    for m in MODELS:
        log = sc.load_log(m)
        col, providers = [], set()
        for r in rows.itertuples():
            rec = log.get((r.question_id, r.country))
            prompt = sc.build_prompt(r.question_text, json.loads(r.options), r.country)
            ok = rec is not None and sc.is_done(rec, prompt)
            col.append(np.array(rec["distribution"]) if ok else None)
            if ok:
                # GPT rows from the direct OpenAI API carry no provider field; the
                # completion via OpenRouter is pinned to OpenAI -> same deployment.
                prov = rec["provider"]
                if m in config.OPENAI_RATERS and prov in (None, "OpenAI"):
                    prov = "OpenAI"
                providers.add(prov)
        if len(providers) > 1:
            raise SystemExit(f"{m}: rows come from several providers {providers}; "
                             "all rows of a model must use one deployment.")
        rows[m] = col
        coverage[m] = int(rows[m].notna().sum())
    common = rows[[all(rows.loc[i, m] is not None for m in MODELS) for i in rows.index]].copy()
    print(f"  Rows: {len(rows):,} | answered per model: {coverage} | common set: {len(common):,}")
    return common.reset_index(drop=True), coverage, len(rows)


# =============================================================================
# ROW-LEVEL METRICS
# =============================================================================

def add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # leave-one-country-out baseline (equal weights)
    loo = [None] * len(df)
    for _, g in df.groupby("question_id"):
        if len(g) < 2:
            continue
        H = np.stack(g.h.values)
        tot = H.sum(0)
        for j, idx in enumerate(g.index):
            loo[idx] = (tot - H[j]) / (len(g) - 1)
    df["loo"] = loo
    df["sim_uniform"] = [1 - js_distance(h, np.ones(len(h)) / len(h)) for h in df.h]
    df["sim_loo"] = [1 - js_distance(h, b) if b is not None else np.nan for h, b in zip(df.h, df.loo)]
    df["hn_human"] = df.h.apply(norm_entropy)
    df["top1_loo"] = [float(np.argmax(h) == np.argmax(b)) if b is not None else np.nan
                      for h, b in zip(df.h, df.loo)]
    for m in MODELS:
        df[f"sim_{m}"] = [1 - js_distance(h, p) for h, p in zip(df.h, df[m])]
        df[f"top1_{m}"] = [float(np.argmax(h) == np.argmax(p)) for h, p in zip(df.h, df[m])]
        df[f"hn_{m}"] = df[m].apply(norm_entropy)
        df[f"egap_{m}"] = df[f"hn_{m}"] - df.hn_human
    return df


# =============================================================================
# CLUSTER BOOTSTRAP (over questions)
# =============================================================================

_BOOT_IDX = {}


def cluster_boot(df: pd.DataFrame, col: str, rng, B: int):
    """
    Cluster bootstrap over questions. All columns evaluated on the same set of
    questions share the SAME resampling indices, so differences between the
    returned bootstrap vectors form a paired bootstrap.
    """
    d = df[["question_id", col]].dropna()
    g = d.groupby("question_id")[col].agg(["sum", "count"])
    s, n = g["sum"].values, g["count"].values
    est = s.sum() / n.sum()
    key = (tuple(g.index), B)
    if key not in _BOOT_IDX:
        _BOOT_IDX[key] = np.random.default_rng(config.RATER_SEED).integers(0, len(g), size=(B, len(g)))
    idx = _BOOT_IDX[key]
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(est), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda x: x[1])
    out, running, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# =============================================================================
# CDP (Chae et al., 2026)
# =============================================================================

def cd(df: pd.DataFrame, col: str, countries=None) -> float:
    vals = []
    d = df if countries is None else df[df.country.isin(countries)]
    for _, g in d.groupby("question_id"):
        if len(g) < 2:
            continue
        P = np.stack(g[col].values)
        c = P.mean(0)
        vals += [jsd2(p, c) for p in P]
    return float(np.mean(vals)) if vals else np.nan


def cdp_table(df: pd.DataFrame) -> dict:
    out = {"overall": {"CD_human": cd(df, "h")}}
    for m in MODELS:
        out["overall"][m] = 100 * cd(df, m) / out["overall"]["CD_human"]
    for region in sorted(df.schwartz_region.unique()):
        if region == "Unassigned":
            continue
        cs = df.loc[df.schwartz_region == region, "country"].unique()
        h = cd(df, "h", cs)
        if np.isnan(h) or h == 0:
            continue
        out[region] = {"CD_human": h, **{m: 100 * cd(df, m, cs) / h for m in MODELS}}
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    tag = "pilot" if args.pilot else "full"
    outdir = config.RESULTS_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PluralValueBench v2 — Stage II metrics ({tag})")
    print("=" * 70)
    unit_tests()
    df, coverage, n_all = load(args.pilot)
    df = add_row_metrics(df)
    rng = np.random.default_rng(config.RATER_SEED)
    B = config.BOOTSTRAP_B
    multi = df[df.loo.notna()]

    res = {"tag": tag, "rows_input": n_all, "rows_common": len(df),
           "questions": int(df.question_id.nunique()), "countries": int(df.country.nunique()),
           "coverage": coverage}

    # ---- Table: fidelity vs baselines --------------------------------------
    main_tab, boots = {}, {}
    for name, col in [("uniform", "sim_uniform")] + [(m, f"sim_{m}") for m in MODELS]:
        est, lo, hi, bs = cluster_boot(df, col, rng, B)
        main_tab[name] = {"similarity": est, "ci": [lo, hi]}
        boots[name] = bs
    for m in MODELS:
        main_tab[m]["top1"] = float(df[f"top1_{m}"].mean())
        main_tab[m]["entropy_gap"] = float(df[f"egap_{m}"].mean())
        main_tab[m]["share_more_dispersed"] = float((df[f"egap_{m}"] > 0).mean())
        main_tab[m]["norm_entropy"] = float(df[f"hn_{m}"].mean())
    res["fidelity_all_rows"] = main_tab
    res["human_norm_entropy"] = float(df.hn_human.mean())

    # on multi-country questions: country-agnostic baseline comparison
    tab_multi = {}
    est, lo, hi, _ = cluster_boot(multi, "sim_loo", rng, B)
    tab_multi["country_agnostic"] = {"similarity": est, "ci": [lo, hi], "top1": float(multi.top1_loo.mean())}
    for m in MODELS:
        est, lo, hi, _ = cluster_boot(multi, f"sim_{m}", rng, B)
        diff = multi[f"sim_{m}"] - multi.sim_loo
        d_est, d_lo, d_hi, _ = cluster_boot(multi.assign(_d=diff), "_d", rng, B)
        tab_multi[m] = {"similarity": est, "ci": [lo, hi], "top1": float(multi[f"top1_{m}"].mean()),
                        "minus_country_agnostic": d_est, "diff_ci": [d_lo, d_hi],
                        "win_rate_vs_country_agnostic": float((diff > 0).mean())}
    res["vs_country_agnostic_multi_country_rows"] = {"n_rows": len(multi), **tab_multi}

    # ---- pairwise model differences (paired cluster bootstrap) -------------
    pv, diffs = {}, {}
    for a, b in combinations(MODELS, 2):
        d = boots[a] - boots[b]
        est = main_tab[a]["similarity"] - main_tab[b]["similarity"]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        diffs[f"{a} - {b}"] = {"diff": est, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        pv[f"{a} - {b}"] = max(float(p), 1.0 / B)
    for k, p in holm(pv).items():
        diffs[k]["p_holm"] = p
    res["pairwise_differences"] = diffs

    # ---- CDP --------------------------------------------------------------
    res["cdp"] = cdp_table(multi)

    # ---- breakdowns -------------------------------------------------------
    def breakdown(mask_fn, keys):
        out = {}
        for k in keys:
            sub = df[mask_fn(k)]
            if len(sub) == 0:
                continue
            sub_multi = sub[sub.loo.notna()]
            out[k] = {"n_rows": len(sub), "n_questions": int(sub.question_id.nunique()),
                      "country_agnostic": float(sub_multi.sim_loo.mean()) if len(sub_multi) else None,
                      **{m: float(sub[f"sim_{m}"].mean()) for m in MODELS},
                      **{f"egap_{m}": float(sub[f"egap_{m}"].mean()) for m in MODELS}}
        return out
    res["by_value"] = breakdown(lambda v: df.labels.apply(lambda l: v in l), config.SCHWARTZ_VALUES)
    res["by_region"] = breakdown(lambda r: df.schwartz_region == r, sorted(df.schwartz_region.unique()))
    res["by_source"] = breakdown(lambda s: df.source == s, sorted(df.source.unique()))

    # ---- inter-rater -------------------------------------------------------
    res["inter_rater_js_distance"] = {f"{a} vs {b}": float(np.mean([js_distance(x, y) for x, y in zip(df[a], df[b])]))
                                      for a, b in combinations(MODELS, 2)}

    # ---- Schwartz orientations (country level) ------------------------------
    cl = df.groupby("country").agg(**{m: (f"sim_{m}", "mean") for m in MODELS},
                                   **{c: (c, "first") for c in ORIENT_COLS},
                                   n=("question_id", "size")).dropna(subset=["or_harmony"])
    cl = cl[cl.n >= 10]
    res["orientation_spearman"] = {"n_countries": len(cl), **{
        m: {c.replace("or_", ""): dict(zip(["rho", "p"], map(float, spearmanr(cl[m], cl[c]))))
            for c in ORIENT_COLS} for m in MODELS}}

    # same correlations for accuracy relative to the country-agnostic baseline: a
    # country close to the cross-country mean is easy for any predictor, so the
    # baseline's country mean is subtracted (rows with a baseline only)
    multi_cl = multi.groupby("country").agg(**{m: (f"sim_{m}", "mean") for m in MODELS},
                                            baseline=("sim_loo", "mean"),
                                            **{c: (c, "first") for c in ORIENT_COLS},
                                            n=("question_id", "size")).dropna(subset=["or_harmony"])
    multi_cl = multi_cl[multi_cl.n >= 10]
    rho_p = lambda x, c: dict(zip(["rho", "p"], map(float, spearmanr(x, multi_cl[c]))))
    res["orientation_spearman_adjusted"] = {
        "n_countries": len(multi_cl),
        "baseline": {c.replace("or_", ""): rho_p(multi_cl.baseline, c) for c in ORIENT_COLS},
        **{m: {c.replace("or_", ""): rho_p(multi_cl[m] - multi_cl.baseline, c) for c in ORIENT_COLS}
           for m in MODELS}}

    # ---- cost / efficiency ------------------------------------------------------
    res["efficiency"] = {m: sc.summarize(sc.select_rows(args.pilot, None), m) for m in MODELS}

    with open(outdir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    df.drop(columns=["h", "loo", *MODELS]).to_csv(outdir / "row_metrics.csv", index=False)

    # ---- console summary ----------------------------------------------------------
    print(f"\n  Common rows {len(df):,} | questions {res['questions']} | countries {res['countries']}")
    print("\n  Fidelity (1 - JS distance), all rows [95% CI]  | top-1 | entropy gap (share >0)")
    print(f"    {'uniform':<16} {main_tab['uniform']['similarity']:.3f} "
          f"[{main_tab['uniform']['ci'][0]:.3f}, {main_tab['uniform']['ci'][1]:.3f}]")
    for m in MODELS:
        t = main_tab[m]
        print(f"    {m:<16} {t['similarity']:.3f} [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}]  "
              f"| {t['top1']:.3f} | {t['entropy_gap']:+.3f} ({t['share_more_dispersed']:.2f})")
    print(f"    human normalised entropy: {res['human_norm_entropy']:.3f}")
    print(f"\n  Multi-country rows (n={len(multi):,}): vs country-agnostic baseline")
    ca = tab_multi["country_agnostic"]
    print(f"    {'country-agnostic':<16} {ca['similarity']:.3f} [{ca['ci'][0]:.3f}, {ca['ci'][1]:.3f}] | top-1 {ca['top1']:.3f}")
    for m in MODELS:
        t = tab_multi[m]
        print(f"    {m:<16} {t['similarity']:.3f}  diff {t['minus_country_agnostic']:+.3f} "
              f"[{t['diff_ci'][0]:+.3f}, {t['diff_ci'][1]:+.3f}]  win-rate {t['win_rate_vs_country_agnostic']:.2f}")
    print("\n  CDP (% of human between-country divergence; <100 = flattening)")
    for k, v in res["cdp"].items():
        print(f"    {k:<32} CD_h={v['CD_human']:.4f}  " + "  ".join(f"{m}={v[m]:.0f}" for m in MODELS))
    print("\n  By value (similarity; country-agnostic baseline in brackets)")
    for k, v in res["by_value"].items():
        ca = f"{v['country_agnostic']:.3f}" if v["country_agnostic"] is not None else "  -  "
        print(f"    {k:<15} n={v['n_rows']:>5} q={v['n_questions']:>3} [{ca}] " +
              " ".join(f"{v[m]:.3f}" for m in MODELS))
    print("\n  By region")
    for k, v in res["by_region"].items():
        ca = f"{v['country_agnostic']:.3f}" if v["country_agnostic"] is not None else "  -  "
        print(f"    {k:<32} n={v['n_rows']:>5} [{ca}] " + " ".join(f"{v[m]:.3f}" for m in MODELS))
    print("\n  Inter-rater JS distance")
    for k, v in res["inter_rater_js_distance"].items():
        print(f"    {k:<36} {v:.3f}")
    print(f"\n  Saved -> {outdir/'metrics.json'}")


if __name__ == "__main__":
    main()
