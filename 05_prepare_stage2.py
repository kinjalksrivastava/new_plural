"""
===============================================================================
PluralValueBench v2 — Stage II, Step 5: Prepare the evaluation input
===============================================================================

Adds country-level metadata to the final Stage I corpus (Q3) and selects the
pilot subset.

1. Schwartz transnational cultural region (Schwartz, 2006; 2008). Eight regions.
   Membership (SCHWARTZ_REGIONS below) follows Schwartz (2008), Ch. 3:
     - East-Central & Baltic and Orthodox East European: listed in the text.
     - All other regions: Figure 3 (co-plot of 77 national groups; colour =
       region), checked against the text.
   Unassigned: countries not sampled by Schwartz, and the three he names as
   located outside their expected region (Turkey, Greek Cyprus, Israel — the
   latter is split into Jewish / Arab samples, our data is one national sample).
2. Schwartz's seven cultural orientation scores (data/raw/schwartz/, see SOURCE.md).
3. Structural-zero flag: a country that was never offered an answer category
   (reduced/recoded scale) shows exact 0% on that option for (almost) every
   question sharing the option set, while other countries use it. Such rows
   measure fieldwork, not opinion, and are excluded from Stage II metrics.
4. Pilot subset: PILOT_N_QUESTIONS questions asked in >= 2 countries, sampled
   round-robin over Schwartz values (so every value is represented), plus the
   WVS-6 expansion items. All countries of a selected question are kept so that
   between-country metrics and the country-agnostic baseline are computable.

Output: data/stage2_input.csv, data/stage2_pilot_question_ids.json,
        data/stage2_prep_report.json
===============================================================================
"""

import json

import numpy as np
import pandas as pd

import config

SCHWARTZ_REGIONS = {
    "West European": [
        "Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Greece",
        "Italy", "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland"],
    "English-speaking": [
        "Australia", "Canada", "Great Britain", "New Zealand", "United States"],
    "Latin American": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Mexico", "Peru", "Venezuela"],
    "East-Central & Baltic European": [
        "Croatia", "Czech Republic", "Estonia", "Hungary", "Latvia", "Poland",
        "Romania", "Slovakia", "Slovenia"],
    "Orthodox East European": [
        "Bosnia and Herzegovina", "Bulgaria", "Georgia", "North Macedonia",
        "Russia", "Serbia", "Ukraine"],
    "Confucian-influenced": [
        "China", "Hong Kong", "Japan", "South Korea", "Taiwan", "Thailand"],
    "South & Southeast Asian": [
        "India", "Indonesia", "Malaysia", "Philippines", "Singapore"],
    "African & Middle Eastern": [
        "Egypt", "Ethiopia", "Ghana", "Iran", "Jordan", "Nigeria", "Senegal",
        "South Africa", "Uganda", "Yemen", "Zimbabwe"],
}
REGION_SOURCE = {
    "East-Central & Baltic European": "Schwartz 2008 text",
    "Orthodox East European": "Schwartz 2008 text",
}
MISFITS = {"Turkey", "Cyprus", "Israel"}

# our canonical country name -> country_long in cult_values_pradadata.csv
ORIENTATION_NAMES = {
    "Argentina": "argentina", "Australia": "austrialia", "Austria": "austria",
    "Belgium": "belgium", "Bolivia": "bolivia", "Bosnia and Herzegovina": "bosniaherz",
    "Brazil": "brazil", "Bulgaria": "bulgaria", "Canada": "canada", "Chile": "chile",
    "China": "china", "Colombia": "colombia", "Croatia": "croatia", "Cyprus": "cyprus",
    "Czech Republic": "czech rep", "Denmark": "denmark", "Egypt": "egypt",
    "Estonia": "estonia", "Ethiopia": "ethiopia", "Finland": "finland", "France": "france",
    "Georgia": "georgia", "Germany": "germany", "Ghana": "ghana", "Greece": "greece",
    "Hong Kong": "hongkong", "Hungary": "hungary", "India": "india",
    "Indonesia": "indonesia", "Iran": "iran", "Italy": "italy", "Japan": "japan",
    "Jordan": "jordan", "South Korea": "korea south", "Latvia": "latvia",
    "North Macedonia": "macedonia", "Malaysia": "malaysia", "Mexico": "mexico",
    "Netherlands": "netherlands", "New Zealand": "newzealand", "Nigeria": "nigeria",
    "Norway": "norway", "Pakistan": "pakistan", "Peru": "peru",
    "Philippines": "philippines", "Poland": "poland", "Portugal": "portugal",
    "Romania": "romania", "Russia": "russia", "Senegal": "senegal", "Serbia": "serbia",
    "Singapore": "singapore", "Slovakia": "slovakia", "Slovenia": "slovenia",
    "South Africa": "south africa", "Spain": "spain", "Sweden": "sweden",
    "Switzerland": "switzerland", "Taiwan": "taiwan", "Thailand": "thailand",
    "Turkey": "turkey", "Uganda": "uganda", "Ukraine": "ukraine",
    "Great Britain": "unitedkingdom", "United States": "unitedstates",
    "Venezuela": "venezuela", "Yemen": "yemen", "Zimbabwe": "zimbabwe",
    # Israel deliberately omitted: Schwartz reports separate Jewish / Arab samples.
}
ORIENTATIONS = {
    "harmony": "harmony", "embedded": "embeddedness", "hierarchy": "hierarchy",
    "mastery": "mastery", "aff_auton": "affective_autonomy",
    "intel_auton": "intellectual_autonomy", "egalitar": "egalitarianism",
}


def assign_regions(rows: pd.DataFrame) -> pd.DataFrame:
    member = {c: r for r, cs in SCHWARTZ_REGIONS.items() for c in cs}
    units = set(rows.country)
    missing = [c for c in member if c not in units]
    assert not missing, f"region list names countries not in the corpus: {missing}"
    rows["schwartz_region"] = rows.country.map(member).fillna("Unassigned")
    rows["region_source"] = np.where(
        rows.country.isin(MISFITS), "unassigned: Schwartz misfit",
        np.where(rows.schwartz_region == "Unassigned", "unassigned: not in Schwartz sample",
                 rows.schwartz_region.map(REGION_SOURCE).fillna("Schwartz 2008 Fig. 3")))
    return rows


def add_orientations(rows: pd.DataFrame) -> pd.DataFrame:
    cv = pd.read_csv(config.SCHWARTZ_DIR / "cult_values_pradadata.csv")
    cv = cv.set_index("country_long")
    unknown = [v for v in ORIENTATION_NAMES.values() if v not in cv.index]
    assert not unknown, f"names not found in Schwartz scores: {unknown}"
    for src, dst in ORIENTATIONS.items():
        rows[f"or_{dst}"] = rows.country.map(
            {c: float(cv.loc[n, src]) for c, n in ORIENTATION_NAMES.items()})
    return rows


def flag_structural_zeros(rows: pd.DataFrame) -> pd.DataFrame:
    """Country never offered an option: 0% on it in >=80% of its >=3 questions sharing
    the option set, while >50% of other countries' rows give it >2%."""
    rows["structural_zero"] = False
    P = rows.human_distribution.apply(lambda s: np.array(json.loads(s)))
    ns = rows.nonsubstantive_idx.apply(json.loads)
    flagged = []
    for opts, g in rows.groupby("options"):
        if g.question_id.nunique() < 3:
            continue
        K = len(json.loads(opts))
        for country, gc in g.groupby("country"):
            if gc.question_id.nunique() < 3:
                continue
            Pc = np.stack(P[gc.index].values)
            others = g.index[g.country != country]
            if len(others) == 0:
                continue
            Po = np.stack(P[others].values)
            for k in range(K):
                if k in ns[gc.index[0]]:
                    continue
                if (Pc[:, k] == 0).mean() >= 0.8 and (Po[:, k] > 0.02).mean() > 0.5:
                    hit = gc.index[Pc[:, k] == 0]
                    rows.loc[hit, "structural_zero"] = True
                    flagged.append({"country": country, "option": json.loads(opts)[k],
                                    "n_rows": int(len(hit))})
    return rows, flagged


def select_pilot(rows: pd.DataFrame) -> list:
    rng = np.random.default_rng(config.RATER_SEED)
    q = rows[~rows.structural_zero].drop_duplicates("question_id")
    q = q[(q.n_countries_for_question >= 2) & (~q.is_none) & (~q.is_expansion)]
    pools = {v: list(rng.permutation(q[q.schwartz_labels.str.contains(f'"{v}"')].question_id))
             for v in config.SCHWARTZ_VALUES}
    chosen = []
    while len(chosen) < config.PILOT_N_QUESTIONS:
        progressed = False
        for v in config.SCHWARTZ_VALUES:
            while pools[v] and pools[v][0] in chosen:
                pools[v].pop(0)
            if pools[v] and len(chosen) < config.PILOT_N_QUESTIONS:
                chosen.append(pools[v].pop(0))
                progressed = True
        if not progressed:
            break
    chosen += sorted(rows.loc[rows.is_expansion, "question_id"].unique())
    return chosen


def main():
    print("=" * 70)
    print("PluralValueBench v2 — Stage II, Step 5: prepare evaluation input")
    print("=" * 70)
    rows = pd.read_csv(config.Q3_ROWS_CSV)
    rows = assign_regions(rows)
    rows = add_orientations(rows)
    rows, flagged = flag_structural_zeros(rows)
    pilot = select_pilot(rows)
    rows["is_pilot"] = rows.question_id.isin(pilot)

    rows.to_csv(config.STAGE2_INPUT_CSV, index=False)
    with open(config.STAGE2_PILOT_IDS, "w") as f:
        json.dump(pilot, f, indent=1)

    units = rows.drop_duplicates("country")
    report = {
        "rows": int(len(rows)),
        "rows_structural_zero_excluded": int(rows.structural_zero.sum()),
        "structural_zero_detail": flagged,
        "countries_per_region": units.schwartz_region.value_counts().to_dict(),
        "row_share_per_region": rows.schwartz_region.value_counts(normalize=True).round(4).to_dict(),
        "countries_with_orientation_scores": int(units.or_harmony.notna().sum()),
        "row_share_with_orientation_scores": round(float(rows.or_harmony.notna().mean()), 4),
        "pilot_questions": len(pilot),
        "pilot_rows": int(rows.is_pilot.sum()),
        "pilot_rows_per_value": {v: int(rows[rows.is_pilot].schwartz_labels.str.contains(f'"{v}"').sum())
                                 for v in config.SCHWARTZ_VALUES},
    }
    with open(config.STAGE2_PREP_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        if k != "structural_zero_detail":
            print(f"  {k}: {v}")
    print(f"  structural zeros: {flagged}")
    print(f"\n  Saved -> {config.STAGE2_INPUT_CSV}")


if __name__ == "__main__":
    main()
