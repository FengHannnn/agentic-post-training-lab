"""Build or re-audit synthetic, balanced v2 preferences without model generation."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_dpo_preferences import protected_cases
from freeze_sft_baseline import verify
from agentic_post_training.data.balanced_preferences import audit_balanced_design, build_balanced_preferences
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, validate_preferences

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baselines/sft-v2-phase3"
FILES = ("train.jsonl", "train_small.jsonl", "development.jsonl")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="构造/审计均衡嵌套偏好数据，不训练、不生成模型回答")
    parser.add_argument("--output", type=Path, default=ROOT / "data/preferences/dpo_v2")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    output = args.output.resolve()
    if not args.audit_only and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    verify(BASELINE)
    baseline_hash = file_hash(BASELINE / "manifest.json")
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(BASELINE / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    old_folder = ROOT / "data/preferences/dpo_v1"
    protected = protected_cases()
    old_hashes = {}
    for name in ("train.jsonl", "development.jsonl"):
        protected += [row.target() for row in load_preferences(old_folder / name)]
        old_hashes[name] = file_hash(old_folder / name)
    protected_hash = hashlib.sha256(json.dumps([asdict(case) for case in protected],
        ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if args.audit_only:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["baseline_manifest_sha256"] != baseline_hash or manifest["protected_v1_data_sha256"] != old_hashes:
            raise ValueError("baseline or protected v1 data changed")
        if manifest["protected_targets_sha256"] != protected_hash:
            raise ValueError("protected targets changed")
        for name in FILES:
            if file_hash(output / name) != manifest["files"][name]:
                raise ValueError(f"preference file changed: {name}")
        rows = {Path(name).stem: load_preferences(output / name) for name in FILES}
    else:
        rows = build_balanced_preferences(baseline_hash)
    design = audit_balanced_design(rows)
    for selected in rows.values():
        for row in selected:
            if (row.provenance["baseline_manifest_sha256"] != baseline_hash
                or row.provenance["rejected_source"] != "synthetic_counterexample"):
                raise ValueError(f"{row.id}: unexpected provenance")
    audit = {size: validate_preferences(rows[split], rows["development"], protected, tokenizer, 192)
             for size, split in (("large", "train"), ("small", "train_small"))}
    # Audit-only also detects drift between saved summaries and current validation.
    if args.audit_only:
        if manifest["audit"] != audit or manifest["design"] != design:
            raise ValueError("saved audit/design summary differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for split, selected in rows.items():
            (output / f"{split}.jsonl").write_text("".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in selected),
                encoding="utf-8", newline="\n")
        manifest = {"dataset_id": "dpo-v2-balanced-synthetic", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "baseline_manifest_sha256": baseline_hash, "protected_v1_data_sha256": old_hashes,
            "max_length": 192, "design": design, "audit": audit,
            "files": {name: file_hash(output / name) for name in FILES},
            "small_subset_seed": 42, "small_families": sorted({row.family_id for row in rows["train_small"]}),
            "protected_case_records": len(protected), "protected_targets_sha256": protected_hash,
            "limitations": ["全为合成负例，不声称来自模型实际错误", "train/dev共享12种措辞与形式组合，不能证明模板外泛化",
                "仅ID、prompt、family和精确AST隔离，不是所有语义等价类隔离", "多条样本共享数字family，不是独立样本",
                "新增数据尚未训练或采集基线生成，不是性能结果", "旧v1与v2同时改变措辞、来源与数据量，不能作纯规模因果对照"]}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "design": design, "audit": audit}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
