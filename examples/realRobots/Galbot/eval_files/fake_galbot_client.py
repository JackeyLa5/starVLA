"""Fake Galbot client for smoke-testing a running StarVLA policy server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


THIS_FILE = Path(__file__).resolve()
STARVLA_DIR = THIS_FILE.parents[4]
if str(STARVLA_DIR) not in sys.path:
    sys.path.insert(0, str(STARVLA_DIR))

from examples.realRobots.Galbot.eval_files.model2galbot_interface import GalbotModelClient


def make_fake_image(width: int, height: int) -> np.ndarray:
    """Create a deterministic RGB test image."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = x[None, :]
    image[..., 1] = y
    image[..., 2] = 127
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host.")
    parser.add_argument("--port", type=int, default=6694, help="Policy server port.")
    parser.add_argument("--unnorm-key", default=None, help="Optional dataset stats key.")
    parser.add_argument("--instruction", default="grasp the object", help="Fake task instruction.")
    parser.add_argument("--width", type=int, default=640, help="Fake head-camera image width.")
    parser.add_argument("--height", type=int, default=360, help="Fake head-camera image height.")
    parser.add_argument("--state-dim", type=int, default=21, help="Fake Galbot state dimension.")
    parser.add_argument("--steps", type=int, default=300, help="Number of inference requests to send.")
    parser.add_argument(
        "--action-refresh-steps",
        type=int,
        default=1,
        help="How often to request a new action chunk. Default 1 sends one server request per step.",
    )
    args = parser.parse_args()

    client = GalbotModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        action_refresh_steps=args.action_refresh_steps,
    )
    image = make_fake_image(width=args.width, height=args.height)
    state = np.zeros((args.state_dim,), dtype=np.float32)

    for step in range(args.steps):
        action = client.step(
            front_head=image,
            state=state,
            instruction=args.instruction,
            step=step,
        )
        preview = np.array2string(action[: min(6, action.shape[0])], precision=4, separator=", ")
        print(f"step={step} action_shape={action.shape} action_head={preview}")


if __name__ == "__main__":
    main()
