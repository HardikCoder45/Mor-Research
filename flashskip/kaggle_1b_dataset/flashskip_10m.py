#!/usr/bin/env python3
"""FlashSkip-10M: a 9,964,832-parameter Chinchilla-scale training run."""

from __future__ import annotations

import json

import flashskip_100m as base


def config() -> base.FlashSkipConfig:
    return base.FlashSkipConfig(
        d_model=160,
        d_ff=360,
        n_layers=10,
        n_q_heads=5,
        n_kv_heads=1,
        attention_layers=(0, 1, 3, 5, 7, 9),
        model_name="FlashSkip-10M",
        architecture_version="10m-v1-2026-09-01",
    )


def main() -> None:
    args = base.make_parser().parse_args()
    cfg = config()
    breakdown = base.exact_parameter_breakdown(cfg)
    print("CONFIG", json.dumps(cfg.to_jsonable(), sort_keys=True), flush=True)
    print("PARAMETER_BREAKDOWN", json.dumps(breakdown, sort_keys=True), flush=True)
    if args.mode == "smoke":
        base.smoke(args, cfg)
    elif args.mode == "prepare":
        base.prepare_data(args, cfg)
    elif args.mode == "train":
        base.train(args, cfg)
    elif args.mode == "evaluate":
        base.evaluate(args, cfg)
    elif args.mode == "verify":
        base.verify(args, cfg)
    elif args.mode == "package":
        base.package(args)


if __name__ == "__main__":
    main()
