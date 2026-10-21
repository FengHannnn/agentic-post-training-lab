"""Verify a transferred bundle without importing torch or loading a model."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "cloud_bundle_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing, changed, or unsafe bundle file: {relative}")
    from freeze_sft_baseline import verify
    verify(ROOT / "artifacts/baselines/sft-v2-phase3")
    print(f"Bundle verified: {len(manifest['files'])} files; source commit {manifest['source_commit']}")
    print("Model loading and GPU inference have NOT been verified by this check.")


if __name__ == "__main__":
    main()
