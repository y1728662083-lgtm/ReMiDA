from __future__ import annotations

import argparse

from protocol_sidnet.config import load_config
from protocol_sidnet.experiments import run_all_protocols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.dataset_root is not None:
        overrides.setdefault("dataset", {})["dataset_root"] = args.dataset_root
    if args.device is not None:
        overrides["device"] = args.device
    cfg = load_config(args.config, overrides=overrides)
    run_all_protocols(cfg)


if __name__ == "__main__":
    main()
