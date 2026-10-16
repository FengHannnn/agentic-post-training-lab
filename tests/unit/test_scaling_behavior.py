from agentic_post_training.data.balanced_preferences import build_balanced_preferences
from agentic_post_training.evaluation.scaling import behavior_record, training_keys, v2_subgroups


def generation(text):
    return {"prediction": text, "raw_prediction": text, "generated_tokens": 30, "hit_token_limit": False}


def test_seen_flags_use_selected_small_dataset_not_full_dataset():
    rows = build_balanced_preferences("fixed")
    small_ids = {row.id for row in rows["train_small"]}
    example = next(row for row in rows["train"] if row.id not in small_ids and row.expected_type == "tool_call")
    seen = {label: training_keys([row.target() for row in rows[split]])
            for label, split in (("small", "train_small"), ("large", "train"))}
    record = behavior_record(example.target(), generation(example.chosen), seen, preference=example)
    assert not record["seen_small_training_prompt"]
    assert not record["seen_small_training_expression"]
    assert record["seen_large_training_prompt"] and record["seen_large_training_expression"]


def test_v2_development_separates_greetings_and_has_24_complete_pairs():
    examples = build_balanced_preferences("fixed")["development"]
    records = [behavior_record(row.target(), generation(row.chosen), {}, preference=row) for row in examples]
    summary = v2_subgroups(records)
    assert summary["arithmetic"]["pair_semantic_pass"] == {"passed": 24, "total": 24}
    assert summary["greetings"]["metrics"]["end_to_end_pass"]["passed"] == 8
    assert len([row for row in records if "pair_id" not in row]) == 8
    # One malformed arithmetic output breaks one pair, not the greeting groups.
    example = examples[0]
    records[0] = behavior_record(example.target(), generation("not JSON"), {}, preference=example)
    assert v2_subgroups(records)["arithmetic"]["pair_semantic_pass"] == {"passed": 23, "total": 24}


def test_numeric_equivalence_preserves_semantic_pair_but_not_strict_ast():
    examples = build_balanced_preferences("fixed")["development"]
    records = [behavior_record(row.target(), generation(row.chosen), {}, preference=row) for row in examples]
    example = examples[0]
    a, b, c = example.provenance["numbers"]
    equivalent = example.chosen.replace(example.expected_expression, f"({b}+{c})*{a}")
    records[0] = behavior_record(example.target(), generation(equivalent), {}, preference=example)
    assert not records[0]["end_to_end_pass"] and records[0]["semantic_pass"]
    assert v2_subgroups(records)["arithmetic"]["pair_semantic_pass"]["passed"] == 24
