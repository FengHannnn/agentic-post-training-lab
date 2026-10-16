import hashlib
import json
from dataclasses import asdict

import pytest

from agentic_post_training.data.balanced_preferences import build_balanced_preferences
from agentic_post_training.data.preferences import load_preference_splits


@pytest.fixture
def dataset(tmp_path):
    rows = build_balanced_preferences("fixed")
    files = {}
    for split, selected in rows.items():
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in selected), encoding="utf-8")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tmp_path, {"files": files}, rows


def test_selected_small_split_is_used_without_changing_development(dataset):
    folder, manifest, rows = dataset
    small = load_preference_splits(folder, manifest, "train_small.jsonl")
    large = load_preference_splits(folder, manifest)
    assert small["train"] == rows["train_small"]
    assert large["train"] == rows["train"]
    assert small["development"] == large["development"] == rows["development"]


@pytest.mark.parametrize("filename", ["development.jsonl", "not_pinned.jsonl"])
def test_unpinned_or_development_file_cannot_be_selected_for_training(dataset, filename):
    folder, manifest, _ = dataset
    with pytest.raises(ValueError, match="training split"):
        load_preference_splits(folder, manifest, filename)


def test_even_unselected_training_file_must_match_manifest(dataset):
    folder, manifest, _ = dataset
    with (folder / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_preference_splits(folder, manifest, "train_small.jsonl")


def test_manifest_paths_cannot_escape_dataset_folder(dataset):
    folder, manifest, _ = dataset
    manifest["files"]["../escape.jsonl"] = "unused"
    with pytest.raises(ValueError, match="inside the dataset folder"):
        load_preference_splits(folder, manifest)
