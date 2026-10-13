"""
PluralValueBench v2 — Schwartz value codebook.

Single source of truth for the value definitions and the labelling question.
The Stage I classifier prompt AND the human annotation sheet are both built
from this file, so humans and the model judge against identical definitions
(required for a meaningful inter-rater agreement study).

Labelling protocol: binary yes/no judgement per value, multi-label, following
Kiesel et al. (2022, ACL) "Identifying the Human Values behind Arguments" and
Mirzakhmedova et al. (2024, LREC-COLING) "The Touché23-ValueEval Dataset".
No probability scores and no threshold.

Defining goals follow Schwartz (1992) and Schwartz (2012), "An Overview of the
Schwartz Theory of Basic Values", Online Readings in Psychology and Culture 2(1).
"""

from config import SCHWARTZ_VALUES

VALUE_DEFINITIONS = {
    "Self-Direction": (
        "Independent thought and action: choosing, creating, exploring. "
        "Freedom, autonomy, choosing one's own goals, curiosity, creativity, "
        "freedom of expression and of personal lifestyle choices."
    ),
    "Stimulation": (
        "Excitement, novelty, and challenge in life. Variety, daring, adventure, "
        "risk-taking for its own sake. (Not mere enjoyment, which is Hedonism.)"
    ),
    "Hedonism": (
        "Pleasure or sensuous gratification for oneself. Enjoying life, having a "
        "good time, leisure, self-indulgence. (Not novelty, which is Stimulation.)"
    ),
    "Achievement": (
        "Personal success through demonstrating competence according to social "
        "standards. Ambition, capability, success at work or school, being "
        "recognised for one's accomplishments."
    ),
    "Power": (
        "Social status and prestige, control or dominance over people and "
        "resources. Authority, wealth, preserving public image, and the power or "
        "dominance of one's group or nation over others."
    ),
    "Security": (
        "Safety, harmony, and stability of society, of relationships, and of self. "
        "National security, social order, law and order, protection from threats, "
        "economic and personal security, health."
    ),
    "Conformity": (
        "Restraint of actions, inclinations, and impulses likely to upset or harm "
        "others and violate social expectations or norms. Obedience, "
        "self-discipline, politeness, following rules. (Current social norms, "
        "not long-standing customs, which is Tradition.)"
    ),
    "Tradition": (
        "Respect, commitment, and acceptance of the customs and ideas that one's "
        "culture or religion provides. Religious devotion, respect for "
        "traditional practices, family and gender customs, moderation, humility."
    ),
    "Benevolence": (
        "Preserving and enhancing the welfare of people with whom one is in "
        "frequent personal contact (the in-group): family, friends, colleagues, "
        "community. Helpfulness, honesty, loyalty, responsibility, trust in close "
        "others. (Close others, not all people, which is Universalism.)"
    ),
    "Universalism": (
        "Understanding, appreciation, tolerance, and protection for the welfare "
        "of ALL people and for nature. Social justice, equality, human rights, "
        "tolerance of different groups, peace, environmental protection."
    ),
}

# The per-value yes/no question (adapted from Kiesel et al., 2022, to survey items)
LABEL_QUESTION = (
    "Is this value at stake in the survey question, i.e. would answering the "
    "question express a position on this value?"
)

LABELING_RULES = [
    "Answer yes or no for each of the ten values independently.",
    "Judge what the QUESTION is about. Do NOT predict how people would answer.",
    "Answer yes only if the value is clearly at stake, not if it is merely "
    "tangential.",
    "A question may engage several values, or none of them.",
]

assert list(VALUE_DEFINITIONS) == SCHWARTZ_VALUES, "codebook and config value lists differ"


def codebook_text() -> str:
    """Plain-text codebook block (used in the prompt and the annotation guide)."""
    lines = ["Schwartz's ten basic human values:"]
    for i, (v, d) in enumerate(VALUE_DEFINITIONS.items(), 1):
        lines.append(f"{i}. {v}: {d}")
    lines.append("")
    lines.append(f"For each value: {LABEL_QUESTION}")
    lines.append("")
    lines.append("Rules:")
    for r in LABELING_RULES:
        lines.append(f"  - {r}")
    return "\n".join(lines)
