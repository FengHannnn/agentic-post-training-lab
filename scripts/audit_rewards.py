"""Write a new reward audit without modifying the source rollout."""

import argparse
import hashlib
import json
from pathlib import Path

from agentic_post_training.rewards.audit import audit_records

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = audit_records([json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()])
    report["input_path"] = str(args.input.resolve())
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    sources = [Path(__file__), *(ROOT / "src/agentic_post_training" / name for name in (
        "rewards/contract.py", "rewards/audit.py", "evaluation/tool_calls.py",
        "environments/calculator.py", "rollouts/records.py", "algorithms/losses.py"))]
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing historical artifacts.
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "groups", "source_sha256")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
