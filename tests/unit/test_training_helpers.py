import pytest

from agentic_post_training.data import SFTExample
from agentic_post_training.trainers.sft_lora import check_split, train_steps


def test_split_rejects_same_question_with_different_ids():
    train = [SFTExample("train", "system", "question", "answer")]
    evaluation = [SFTExample("eval", "system", "question", "other answer")]
    with pytest.raises(ValueError, match="prompts"):
        check_split(train, evaluation)


def test_split_rejects_overlapping_ids():
    with pytest.raises(ValueError, match="IDs"):
        check_split([SFTExample("same", "s", "a", "x")], [SFTExample("same", "s", "b", "y")])


def test_training_rejects_empty_data():
    with pytest.raises(ValueError, match="non-empty"):
        train_steps(None, [], pad_token_id=0)
