"""Run the frozen r16 learning-rate study."""
import argparse
from pathlib import Path

from grpo_common import ROOT
from run_v2_grpo_dose import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grpo-sql-lr-study-v1.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.resume)
