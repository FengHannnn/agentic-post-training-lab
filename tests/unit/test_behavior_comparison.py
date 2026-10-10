import pytest

from agentic_post_training.evaluation.comparison import compare_records, transition


def row(identifier, passed):
    return dict(id=identifier, system="system", prompt=identifier, category="math",
        expected_type="tool_call", expected_expression="2*3", expected_result=6,
        prediction="correct" if passed else "wrong", failure=None if passed else "wrong_arguments",
        tool_result=6 if passed else 5, semantic_pass=passed, end_to_end_pass=passed,
        json_valid=True, schema_valid=True, route_correct=True, arguments_valid=True,
        arguments_correct=passed, tool_execution_success=True, result_correct=passed, hit_token_limit=False)


def test_compare_aligns_ids_and_counts_regressions():
    sft = [row("a", True), row("b", False), row("c", True), row("d", False)]
    dpo = [row("d", False), row("c", True), row("b", True), row("a", False)]
    result, changes = compare_records(sft, dpo)
    assert result["transitions"]["end_to_end_pass"] == {
        "improved": 1, "regressed": 1, "both_pass": 1, "both_fail": 1}
    assert changes[0]["end_to_end_pass"] == "regressed"
    assert result["prediction_changed_count"] == 2


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique IDs"):
        compare_records([row("a", True)] * 2, [row("a", True)])


def test_mismatched_targets_are_rejected():
    left, right = row("a", True), row("a", True)
    right["expected_expression"] = "3*2"
    with pytest.raises(ValueError, match="targets differ"):
        compare_records([left], [right])


def test_semantic_improvement_can_differ_from_strict():
    left, right = row("a", False), row("a", False)
    right["semantic_pass"] = True
    result, _ = compare_records([left], [right])
    assert result["transitions"]["semantic_pass"]["improved"] == 1
    assert result["transitions"]["end_to_end_pass"]["both_fail"] == 1
