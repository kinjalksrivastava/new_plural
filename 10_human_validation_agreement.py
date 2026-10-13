"""
===============================================================================
PluralValueBench v2 — Stage I, Step 10: Human validation agreement
===============================================================================

Scores the two completed annotation workbooks against each other and against
the classifier labels for the sample drawn in step 9.

- Krippendorff's alpha (nominal, two coders, no missing data) between the two
  annotators, per value and over all decisions.
- Cohen's kappa between each annotator and the classifier, per value and over
  all decisions.
- Share of the labels that both annotators assign which the classifier also
  assigns, and the mean number of values per question for each labeller.

Every value cell must hold yes or no; the script stops otherwise.

Output: human_validation/agreement_report.json, agreement_table.tex
        (the table is also written to paper/agreement_table.tex)
===============================================================================
"""

import json

import numpy as np
import pandas as pd

import config

DIR = config.ROOT_DIR / "human_validation"
VALUES = config.SCHWARTZ_VALUES
FILES = {"A": DIR / "annotator_A_completed.xlsx", "B": DIR / "annotator_B_completed.xlsx"}


def load(path, sheet) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)[["item_id"] + VALUES]
    for v in VALUES:
        df[v] = df[v].astype(str).str.strip().str.lower()
    bad = int((~df[VALUES].isin(["yes", "no"])).values.sum())
    if bad or not df.item_id.is_unique:
        raise SystemExit(f"{path.name}: {bad} cells are not yes/no, or item ids repeat")
    return df.set_index("item_id").sort_index().eq("yes").astype(int)


def cohen_kappa(x, y) -> float:
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def krippendorff_alpha(x, y) -> float:
    n = len(x)
    p = float(np.concatenate([x, y]).mean())
    expected = 2 * p * (1 - p) * (2 * n) / (2 * n - 1)
    return 1 - float((x != y).mean()) / expected if expected > 0 else float("nan")


def main():
    a, b = load(FILES["A"], "Annotation"), load(FILES["B"], "Annotation")
    m = load(DIR / "model_labels.xlsx", "Model labels")
    assert list(a.index) == list(b.index) == list(m.index), "item ids differ between files"

    per_value = {}
    for v in VALUES:
        x, y, z = a[v].values, b[v].values, m[v].values
        per_value[v] = {"yes_A": int(x.sum()), "yes_B": int(y.sum()), "yes_model": int(z.sum()),
                        "agreement_AB": float((x == y).mean()), "alpha_AB": krippendorff_alpha(x, y),
                        "kappa_A_model": cohen_kappa(x, z), "kappa_B_model": cohen_kappa(y, z)}
    A, B, M = a.values.ravel(), b.values.ravel(), m.values.ravel()
    both = (A == 1) & (B == 1)
    report = {
        "n_questions": len(a), "n_decisions": int(A.size),
        "alpha_AB_all_decisions": krippendorff_alpha(A, B),
        "alpha_AB_mean_over_values": float(np.mean([r["alpha_AB"] for r in per_value.values()])),
        "agreement_AB_all_decisions": float((A == B).mean()),
        "kappa_A_model_all_decisions": cohen_kappa(A, M), "kappa_B_model_all_decisions": cohen_kappa(B, M),
        "kappa_A_model_mean_over_values": float(np.mean([r["kappa_A_model"] for r in per_value.values()])),
        "kappa_B_model_mean_over_values": float(np.mean([r["kappa_B_model"] for r in per_value.values()])),
        "labels_both_annotators": int(both.sum()),
        "share_of_those_also_model": float(M[both].mean()),
        "model_labels_with_at_least_one_annotator": float(((A == 1) | (B == 1))[M == 1].mean()),
        "model_labels": int(M.sum()),
        "model_labels_with_neither_annotator": int(((A == 0) & (B == 0) & (M == 1)).sum()),
        "model_no_when_both_annotators_no": float((1 - M[(A == 0) & (B == 0)]).mean()),
        "values_per_question": {"A": float(a.sum(axis=1).mean()), "B": float(b.sum(axis=1).mean()),
                                "model": float(m.sum(axis=1).mean())},
        "per_value": per_value,
    }
    (DIR / "agreement_report.json").write_text(json.dumps(report, indent=2))

    lines = [r"\begin{tabular}{lrrrccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Questions labeled yes} & $\alpha$ & $\kappa$ & $\kappa$ \\",
             r"Value & A & B & Model & A and B & A and model & B and model \\", r"\midrule"]
    for v, r in per_value.items():
        lines.append(f"{v} & {r['yes_A']} & {r['yes_B']} & {r['yes_model']} & {r['alpha_AB']:.2f} & "
                     f"{r['kappa_A_model']:.2f} & {r['kappa_B_model']:.2f} \\\\")
    lines += [r"\midrule",
              f"Mean over values & & & & {report['alpha_AB_mean_over_values']:.2f} & "
              f"{report['kappa_A_model_mean_over_values']:.2f} & {report['kappa_B_model_mean_over_values']:.2f} \\\\",
              f"All decisions & {int(A.sum())} & {int(B.sum())} & {int(M.sum())} & {report['alpha_AB_all_decisions']:.2f} & "
              f"{report['kappa_A_model_all_decisions']:.2f} & {report['kappa_B_model_all_decisions']:.2f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    for folder in (DIR, config.ROOT_DIR / "paper"):          # the paper inputs its own copy
        (folder / "agreement_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_value"}, indent=2))


if __name__ == "__main__":
    main()
