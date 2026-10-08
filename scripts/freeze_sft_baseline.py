"""Snapshot an SFT baseline, or verify an existing snapshot without training."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = "artifacts/checkpoints/sft-lora-20261008-202445-a3d10354"
EVALUATIONS = {
    "tool_calls": "artifacts/evaluations/tool-calls-20261008-203842-cc578a24",
    "brackets": "artifacts/evaluations/bracket-diagnostic-20261008-204902-76e6b53b",
    "spacing_control": "artifacts/evaluations/bracket-diagnostic-20261008-210057-380ef02a",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(baseline):
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = baseline / item["snapshot"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot missing or changed: {path}")
    for item in manifest["base_cache_files"]:
        path = ROOT / item["source"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Base cache missing or changed: {path}")
    print(f"Verified {len(manifest['files'])} snapshot files and "
          f"{len(manifest['base_cache_files'])} base cache files: {baseline}")


def main():
    parser = argparse.ArgumentParser(description="冻结 v2 SFT 基线或校验已有快照")
    parser.add_argument("--verify", type=Path, help="校验含 manifest.json 的基线目录")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
        return

    summary = json.loads((ROOT / RUN / "summary.json").read_text(encoding="utf-8"))
    revision = summary["resolved_revision"]
    if summary["steps_completed"] != 288 or not summary["reload_logits_match"]:
        raise ValueError("Unexpected or incomplete v2 baseline")
    baseline = ROOT / "artifacts/baselines/sft-v2-phase3"
    if baseline.exists():
        raise FileExistsError(f"Baseline already exists; use --verify: {baseline}")

    sources = [(path, Path("checkpoint") / path.relative_to(ROOT / RUN))
               for path in sorted((ROOT / RUN).rglob("*")) if path.is_file()]
    for label, relative in EVALUATIONS.items():
        sources += [(path, Path("evaluations") / label / path.relative_to(ROOT / relative))
                    for path in sorted((ROOT / relative).rglob("*")) if path.is_file()]
    for folder in ("configs", "data/training", "data/eval", "src", "scripts", "tests"):
        sources += [(path, Path("project") / path.relative_to(ROOT))
                    for path in sorted((ROOT / folder).rglob("*"))
                    if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"]
    for name in ("pyproject.toml", "PROJECT_HANDOFF.md", "README.md"):
        sources.append((ROOT / name, Path("project") / name))

    cache = ROOT / ".cache/huggingface/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots" / revision
    base_files = [path for path in sorted(cache.rglob("*")) if path.is_file()]
    if not base_files or not any(path.suffix == ".safetensors" for path in base_files):
        raise ValueError("Pinned base weights are missing from the local cache")
    for source, _ in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    baseline.mkdir(parents=True, exist_ok=False)
    files = []
    for source, relative in sources:
        destination = baseline / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256(source)
        if sha256(destination) != digest:
            raise ValueError(f"Copy verification failed: {source}")
        files.append({"source": source.relative_to(ROOT).as_posix(),
                      "snapshot": relative.as_posix(), "bytes": source.stat().st_size,
                      "sha256": digest})

    metrics = {}
    for label, relative in EVALUATIONS.items():
        result = json.loads((ROOT / relative / "summary.json").read_text(encoding="utf-8"))
        metrics[label] = result["lora"] if label == "tool_calls" else result["v2"]
    manifest = {
        "baseline_id": "sft-v2-phase3",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "source_run": RUN, "model": summary["config"]["model"],
        "resolved_revision": revision, "training_summary": summary,
        "evaluation_role": "development_diagnostics_not_final_test",
        "generation": {"device": "cpu", "dtype": "float32", "attention": "eager",
                       "do_sample": False, "num_beams": 1, "max_new_tokens": 64,
                       "seed": 42, "cpu_threads": 2},
        "dpo_initialization": {"policy": "checkpoint/adapter",
                               "reference": "checkpoint/adapter",
                               "reference_requires_grad": False},
        "metrics": metrics, "files": files,
        "base_cache_files": [{"source": path.relative_to(ROOT).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for path in base_files],
        "base_weights_copied": False,
    }
    (baseline / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    verify(baseline)


if __name__ == "__main__":
    main()
