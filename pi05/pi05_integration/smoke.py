"""Load a real checkpoint and run vanilla plus RTC-guided inference without a socket."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .rtc import align_action_prefix, make_prefix_weights
from .server import Pi05Server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_lowvar_v2_lora_h32")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--delay-steps", type=int, default=1)
    args = parser.parse_args()

    server = Pi05Server(args.config, args.checkpoint, num_steps=args.num_steps)
    observation = {
        "image_primary": np.zeros((256, 256, 3), np.uint8),
        "image_wrist": np.zeros((128, 128, 3), np.uint8),
        "proprio": np.zeros(10, np.float32),
        "language": "open the drawer",
        "embodiment_action_dim": 8,
        "sample_seed": 123,
    }
    vanilla = server.infer(observation)["actions"]
    prefix, valid_horizon = align_action_prefix(
        vanilla, executed_steps=args.execution_horizon, horizon=server.action_horizon
    )
    weights = make_prefix_weights(
        server.action_horizon,
        prefix_change_start=args.delay_steps,
        prefix_horizon=valid_horizon,
    )
    guided = server.infer({
        **observation,
        "sample_seed": 124,
        "action_prefix": prefix,
        "action_prefix_weights": weights,
    })["actions"]
    print(json.dumps({
        "vanilla_shape": list(vanilla.shape),
        "guided_shape": list(guided.shape),
        "vanilla_finite": bool(np.isfinite(vanilla).all()),
        "guided_finite": bool(np.isfinite(guided).all()),
        "prefix_horizon": valid_horizon,
    }))


if __name__ == "__main__":
    main()
