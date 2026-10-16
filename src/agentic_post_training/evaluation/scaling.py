"""Metadata for selected training sets and paired v2 preference development."""

from dataclasses import asdict

from agentic_post_training.data.preferences import semantic_pass
from agentic_post_training.environments.calculator import expression_structure
from agentic_post_training.evaluation.brackets import diagnostic_summary
from agentic_post_training.evaluation.tool_calls import score_output, summarize

GENERATION_FIELDS = ("prediction", "raw_prediction", "generated_tokens", "hit_token_limit")


def training_keys(cases):
    return ({(case.system, case.prompt) for case in cases},
            {expression_structure(case.expected_expression) for case in cases if case.expected_type == "tool_call"})


def behavior_record(case, generation, seen, *, preference=None, legacy_paired=False, source="fresh"):
    score = score_output(generation["prediction"], case)
    row = {**asdict(case), **{key: generation[key] for key in GENERATION_FIELDS}, **score,
           "semantic_pass": semantic_pass(score, case.expected_type), "generation_source": source}
    for label, (prompts, expressions) in seen.items():
        row[f"seen_{label}_training_prompt"] = (case.system, case.prompt) in prompts
        row[f"seen_{label}_training_expression"] = (case.expected_type == "tool_call"
            and expression_structure(case.expected_expression) in expressions)
    if legacy_paired:
        number_set, wording, form = case.category.split("/")
        row.update(number_set=number_set, wording=wording, form=form,
                   pair_id=f"{number_set}/{wording}/{form.split('_')[0]}")
    if preference is not None:
        row.update(number_set=preference.family_id, wording=preference.provenance["wording"], form=preference.category)
        if case.expected_type == "tool_call":
            row["pair_id"] = preference.provenance["contrast_id"]
    return row


def v2_subgroups(records):
    arithmetic = [row for row in records if row["expected_type"] == "tool_call"]
    greetings = [row for row in records if row["expected_type"] == "final"]
    return {"arithmetic": diagnostic_summary(arithmetic), "greetings": summarize(greetings)}
