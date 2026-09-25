"""Entry point.

  python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/base.yaml
  python .../run.py --config .../smoke.yaml --stage make_smoke
  python .../run.py --config .../base.yaml --stage train predict --set run_name=v2 model.params.num_leaves=127

Run from the project root (the folder containing dataset/ and utils/).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

STAGES = ["prepare", "block", "features", "train", "predict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", nargs="+", default=["all"], choices=["all", "make_smoke"] + STAGES)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. blocking.top_n=40")
    args = ap.parse_args()
    pipe = Pipeline(load_config(args.config, args.set))
    for stage in STAGES if args.stage == ["all"] else args.stage:
        pipe.log(f"=== stage: {stage}")
        getattr(pipe, stage)()


if __name__ == "__main__":
    main()
