from dataclasses import replace

import pytest

from agentic_post_training.data.preferences import PreferenceExample, validate_preferences
from agentic_post_training.data.demo_tokenizer import DemoChatTokenizer
from agentic_post_training.evaluation.tool_calls import EvalCase, SYSTEM_PROMPT


def pair(identifier, family, prompt, expression, result, rejected):
    return PreferenceExample(identifier, family, SYSTEM_PROMPT, prompt,
        '{"type":"tool_call","name":"calculator","arguments":{"expression":"' + expression + '"}}',
        '{"type":"tool_call","name":"calculator","arguments":{"expression":"' + rejected + '"}}',
        "arithmetic", "tool_call", expression, result)


@pytest.fixture
def rows():
    return ([pair("train", "f1", "Multiply two by three.", "2*3", 6, "2+3")],
            [pair("dev", "f2", "Multiply four by five.", "4*5", 20, "4+5")])


def test_valid_pairs_have_response_only_tokens(rows):
    result = validate_preferences(*rows, [], DemoChatTokenizer(), 512)
    assert result["train"]["pairs"] == 1
    assert result["development"]["tokens"]["rejected"]["min_response"] > 0


def test_equivalent_expression_is_not_a_rejected_error(rows):
    train, dev = rows
    equivalent = train[0].chosen.replace("2*3", "3*2")
    with pytest.raises(ValueError, match="semantically valid"):
        validate_preferences([replace(train[0], rejected=equivalent)], dev, [])


def test_wrong_chosen_is_rejected(rows):
    train, dev = rows
    with pytest.raises(ValueError, match="chosen"):
        validate_preferences([replace(train[0], chosen="{}")], dev, [])


def test_family_isolation(rows):
    train, dev = rows
    with pytest.raises(ValueError, match="family overlap"):
        validate_preferences(train, [replace(dev[0], family_id="f1")], [])


def test_protected_expression_overlap_even_with_new_prompt(rows):
    protected = [EvalCase("protected", "math", "A different wording", "tool_call", "(2*3)", 6)]
    with pytest.raises(ValueError, match="expression overlap"):
        validate_preferences(*rows, protected)


def test_overlength_rejected_is_not_truncated(rows):
    with pytest.raises(ValueError, match="exceeds max_length"):
        validate_preferences(*rows, [], DemoChatTokenizer(), 10)
