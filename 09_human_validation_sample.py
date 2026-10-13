"""
===============================================================================
PluralValueBench v2 — Stage I, Step 9: Human validation sample
===============================================================================

Draws the sample for the human agreement check of the Schwartz value labels and
writes the annotation workbooks.

1. Sample: N_SAMPLE questions from the classifier-labelled questions of Q3
   (the two WVS-6 items are labelled by design and are excluded). Stratified:
   going from the rarest to the most common value, questions that the
   classifier labelled with the value are added until the sample holds at least
   MIN_PER_VALUE of them; the rest is filled at random. Fixed seed.
2. Two annotator workbooks with the same items in different random order. Each
   has an instruction sheet built from codebook.py (the definitions and rules
   the classifier saw) and an annotation sheet with a yes/no cell per value.
   The classifier labels are NOT in these files.
3. One workbook with the classifier labels for the same items (not for the
   annotators).

Output: human_validation/sample_questions.csv, annotator_A.xlsx,
        annotator_B.xlsx, model_labels.xlsx, sample_report.json
===============================================================================
"""

import json

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import codebook
import config

N_SAMPLE = 150
MIN_PER_VALUE = 15
SEED = config.CLASSIFIER_SEED
OUT_DIR = config.ROOT_DIR / "human_validation"
VALUES = config.SCHWARTZ_VALUES

HEADER_FILL = PatternFill("solid", fgColor="DDE7F3")
WRAP = Alignment(wrap_text=True, vertical="top")


def draw_sample(questions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pool = questions[questions.label_source == "classifier"].copy()
    pool["labels"] = pool.schwartz_labels.map(json.loads)
    counts = {v: int(pool.labels.map(lambda l: v in l).sum()) for v in VALUES}
    chosen = []
    for value in sorted(VALUES, key=counts.get):
        have = sum(value in pool.labels[i] for i in chosen)
        candidates = [i for i in pool.index if value in pool.labels[i] and i not in chosen]
        need = max(0, MIN_PER_VALUE - have)
        chosen += list(rng.choice(candidates, size=min(need, len(candidates)), replace=False))
    rest = [i for i in pool.index if i not in chosen]
    chosen += list(rng.choice(rest, size=N_SAMPLE - len(chosen), replace=False))
    sample = pool.loc[rng.permutation(chosen)].reset_index(drop=True)
    sample.insert(0, "item_id", [f"H{i:03d}" for i in range(1, len(sample) + 1)])
    return sample


def options_text(options_json: str) -> str:
    return "\n".join(f"- {o}" for o in json.loads(options_json))


def style_header(ws, widths):
    for col, width in enumerate(widths, start=1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "D2"


def instruction_sheet(ws):
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 110
    lines = [
        ("Task", "For every survey question in the sheet 'Annotation', decide for each of the ten "
                 "Schwartz values whether it applies. Select yes or no in every value cell."),
        ("Question to ask", codebook.LABEL_QUESTION),
    ]
    lines += [(f"Rule {i}", rule) for i, rule in enumerate(codebook.LABELING_RULES, start=1)]
    lines += [
        ("Work alone", "Do not discuss the items with the other annotator and do not use an AI assistant."),
        ("Comment", "Use the comment column for items you find unclear. Still give a yes or no for every value."),
        ("", ""),
        ("Value", "Definition"),
    ]
    lines += [(v, codebook.VALUE_DEFINITIONS[v]) for v in VALUES]
    for r, (a, b) in enumerate(lines, start=1):
        ws.cell(row=r, column=1, value=a).font = Font(bold=True)
        ws.cell(row=r, column=2, value=b).alignment = WRAP
        ws.cell(row=r, column=1).alignment = WRAP


def annotator_workbook(sample: pd.DataFrame, path):
    wb = Workbook()
    instruction_sheet(wb.active)
    ws = wb.create_sheet("Annotation")
    ws.append(["item_id", "question", "answer options"] + VALUES + ["comment"])
    style_header(ws, [9, 60, 34] + [13] * len(VALUES) + [30])
    for row in sample.itertuples():
        ws.append([row.item_id, row.question_text, options_text(row.options)])
    first, last = get_column_letter(4), get_column_letter(3 + len(VALUES))
    yes_no = DataValidation(type="list", formula1='"yes,no"', allow_blank=True, showErrorMessage=True,
                            errorTitle="yes or no", error="Please select yes or no.")
    ws.add_data_validation(yes_no)
    yes_no.add(f"{first}2:{last}{len(sample) + 1}")
    for cells in ws.iter_rows(min_row=2):
        for cell in cells:
            cell.alignment = WRAP
    wb.active = wb.index(ws)
    wb.save(path)


def model_workbook(sample: pd.DataFrame, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Model labels"
    ws.append(["item_id", "question", "answer options"] + VALUES + ["question_id", "source"])
    style_header(ws, [9, 60, 34] + [13] * len(VALUES) + [12, 9])
    for row in sample.itertuples():
        ws.append([row.item_id, row.question_text, options_text(row.options)]
                  + ["yes" if v in row.labels else "no" for v in VALUES]
                  + [row.question_id, row.source])
    for cells in ws.iter_rows(min_row=2):
        for cell in cells:
            cell.alignment = WRAP
    wb.save(path)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    questions = pd.read_csv(config.Q3_QUESTIONS_CSV, keep_default_na=False)
    sample = draw_sample(questions)

    sample[["item_id", "question_id", "source", "question_text", "options", "schwartz_labels"]].to_csv(
        OUT_DIR / "sample_questions.csv", index=False)
    rng = np.random.default_rng(SEED + 1)
    for name in ["A", "B"]:
        order = sample.iloc[rng.permutation(len(sample))]
        annotator_workbook(order, OUT_DIR / f"annotator_{name}.xlsx")
    model_workbook(sample, OUT_DIR / "model_labels.xlsx")

    report = {
        "n_sample": len(sample), "seed": SEED, "min_per_value": MIN_PER_VALUE,
        "classifier_model": config.CLASSIFIER_MODEL_ID,
        "by_source": sample.source.value_counts().to_dict(),
        "no_value": int((sample.labels.map(len) == 0).sum()),
        "classifier_yes_in_sample": {v: int(sample.labels.map(lambda l: v in l).sum()) for v in VALUES},
        "classifier_yes_in_corpus": {v: int(questions.schwartz_labels.map(lambda s: v in json.loads(s)).sum())
                                     for v in VALUES},
    }
    (OUT_DIR / "sample_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
