#!/usr/bin/env python3
"""FlashSkip-10M Recursive: paper-style Middle-Cycle weight sharing at ~10M parameters.

Inspired by Mixture-of-Recursions (arXiv:2507.10524), this variant keeps the
same ten unique FlashSkip-10M blocks and parameter count, but executes the
middle block stack recursively. With recursion_steps=2 the effective depth is
1 + 8*2 + 1 = 18 layers while blocks 1..8 reuse the exact same parameters on
both passes.

This first fast experiment deliberately uses fixed Middle-Cycle recursion
rather than adaptive token routing so it preserves the existing highly-tested
training loop and gives a clean recursion-vs-baseline comparison.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

import flashskip_100m as base


@dataclass
class RecursiveFlashSkipConfig(base.FlashSkipConfig):
    recursion_steps: int = 2
    shared_start_layer: int = 1
    shared_end_layer: int = 9  # exclusive; layers 1..8 form the recursive middle cycle
    recursion_strategy: str = "middle-cycle"
    paper_reference: str = "arXiv:2507.10524"

    @property
    def effective_layers(self) -> int:
        middle = self.shared_end_layer - self.shared_start_layer
        prefix = self.shared_start_layer
        suffix = self.n_layers - self.shared_end_layer
        return prefix + middle * self.recursion_steps + suffix

    def validate_recursion(self) -> None:
        if self.recursion_steps < 1:
            raise ValueError("recursion_steps must be >= 1")
        if not (0 <= self.shared_start_layer < self.shared_end_layer <= self.n_layers):
            raise ValueError(
                f"invalid shared range [{self.shared_start_layer}, {self.shared_end_layer}) "
                f"for n_layers={self.n_layers}"
            )
        if self.recursion_strategy != "middle-cycle":
            raise ValueError(f"unsupported recursion_strategy={self.recursion_strategy!r}")


class RecursiveFlashSkipLM(base.FlashSkipLM):
    """FlashSkip LM with true shared-weight Middle-Cycle recursion."""

    def __init__(self, cfg: RecursiveFlashSkipConfig):
        cfg.validate_recursion()
        super().__init__(cfg)
        self.cfg = cfg

        # The base initializer scales residual output projections for physical
        # depth. Re-scale for the deeper unrolled computation to keep residual
        # magnitudes stable when the middle stack is revisited.
        residual_std = 0.02 / math.sqrt(2 * cfg.effective_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)
            if block.has_attention:
                torch.nn.init.normal_(block.attn.o.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds configured max {self.cfg.max_seq_len}")

        x = self.tok_embeddings(input_ids)
        cos, sin = base.rope_cos_sin(
            t, self.cfg.head_dim, self.cfg.rope_theta, x.device, x.dtype
        )

        # Unique prefix (for this config, block 0).
        for idx in range(self.cfg.shared_start_layer):
            x = self.blocks[idx](x, cos, sin)

        # Middle-Cycle recursion: the same Module objects/weights are reused on
        # every pass. No duplicate middle blocks are constructed.
        for _recursion in range(self.cfg.recursion_steps):
            for idx in range(self.cfg.shared_start_layer, self.cfg.shared_end_layer):
                x = self.blocks[idx](x, cos, sin)

        # Unique suffix (for this config, block 9).
        for idx in range(self.cfg.shared_end_layer, self.cfg.n_layers):
            x = self.blocks[idx](x, cos, sin)

        x = self.final_norm(x)
        return base.F.linear(x, self.tok_embeddings.weight)


def config() -> RecursiveFlashSkipConfig:
    return RecursiveFlashSkipConfig(
        d_model=160,
        d_ff=360,
        n_layers=10,
        n_q_heads=5,
        n_kv_heads=1,
        attention_layers=(0, 1, 3, 5, 7, 9),
        recursion_steps=2,
        shared_start_layer=1,
        shared_end_layer=9,
        recursion_strategy="middle-cycle",
        model_name="FlashSkip-10M-Recursive-MiddleCycle-Nr2",
        architecture_version="10m-rec-middle-cycle-nr2-v1-2026-09-03",
    )


def install_into_base() -> None:
    """Make the existing trainer/evaluator instantiate the recursive classes."""
    base.FlashSkipConfig = RecursiveFlashSkipConfig
    base.FlashSkipLM = RecursiveFlashSkipLM


def recursion_summary(cfg: RecursiveFlashSkipConfig) -> dict[str, object]:
    middle = list(range(cfg.shared_start_layer, cfg.shared_end_layer))
    executed = (
        list(range(cfg.shared_start_layer))
        + middle * cfg.recursion_steps
        + list(range(cfg.shared_end_layer, cfg.n_layers))
    )
    return {
        "strategy": cfg.recursion_strategy,
        "recursion_steps": cfg.recursion_steps,
        "shared_middle_layers": middle,
        "effective_layer_sequence": executed,
        "unique_layers": cfg.n_layers,
        "effective_layers": cfg.effective_layers,
        "unique_parameters": base.exact_parameter_breakdown(cfg)["total"],
    }


def snapshot_entrypoint(output_dir: str) -> None:
    try:
        src_dir = Path(output_dir) / "source"
        src_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), src_dir / Path(__file__).name)
    except Exception:
        pass


def main() -> None:
    args = base.make_parser().parse_args()
    cfg = config()
    install_into_base()
    breakdown = base.exact_parameter_breakdown(cfg)
    print("CONFIG", json.dumps(cfg.to_jsonable(), sort_keys=True), flush=True)
    print("RECURSION", json.dumps(recursion_summary(cfg), sort_keys=True), flush=True)
    print("PARAMETER_BREAKDOWN", json.dumps(breakdown, sort_keys=True), flush=True)

    if args.mode == "smoke":
        base.smoke(args, cfg)
    elif args.mode == "prepare":
        base.prepare_data(args, cfg)
    elif args.mode == "train":
        snapshot_entrypoint(args.output_dir)
        base.train(args, cfg)
    elif args.mode == "evaluate":
        base.evaluate(args, cfg)
    elif args.mode == "verify":
        base.verify(args, cfg)
    elif args.mode == "package":
        base.package(args)


if __name__ == "__main__":
    main()
