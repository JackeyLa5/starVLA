"""Galbot env-side adapter for the StarVLA websocket policy server."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class GalbotModelClient:
    """Thin Galbot policy client.

    The StarVLA websocket server returns already-unnormalized actions. This
    client only handles image resizing, action chunk caching, and request shape.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 6694,
        unnorm_key: Optional[str] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_refresh_steps: int | None = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        meta = self.client.get_server_metadata()
        self.server_metadata = meta
        self.action_chunk_size = int(meta["action_chunk_size"])
        self.action_refresh_steps = int(action_refresh_steps or self.action_chunk_size)
        if self.action_refresh_steps < 1:
            raise ValueError(f"action_refresh_steps must be >= 1, got {self.action_refresh_steps}.")
        if self.action_refresh_steps > self.action_chunk_size:
            raise ValueError(
                f"action_refresh_steps={self.action_refresh_steps} cannot exceed "
                f"action_chunk_size={self.action_chunk_size}."
            )
        self.raw_actions: np.ndarray | None = None
        self.task_description: str | None = None
        print(
            "[GalbotModelClient] "
            f"unnorm_key={unnorm_key}, action_chunk_size={self.action_chunk_size}, "
            f"action_refresh_steps={self.action_refresh_steps}, "
            f"server_meta={meta}"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = str(task_description)
        self.raw_actions = None

    def step(
        self,
        *,
        front_head: np.ndarray,
        left_wrist: np.ndarray | None = None,
        right_wrist: np.ndarray | None = None,
        state: np.ndarray,
        instruction: str,
        step: int,
    ) -> np.ndarray:
        if instruction != self.task_description:
            self.reset(instruction)

        if step % self.action_refresh_steps == 0 or self.raw_actions is None:
            example = {
                "image": [
                    self._prepare_rgb(front_head),
                ],
                "lang": str(instruction),
                "state": np.asarray(state, dtype=np.float32).reshape(1, -1),
            }
            request = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(request)
            data = _response_data(response)
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 1:
                raise ValueError(f"Expected actions shape (1, T, D), got {actions.shape}.")
            self.raw_actions = actions[0]

        action = np.asarray(self.raw_actions[step % self.action_refresh_steps], dtype=np.float32)
        if action.shape != (21,):
            raise ValueError(f"Expected Galbot 21-d action, got {action.shape}.")
        return action

    def _prepare_rgb(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {arr.shape}.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)


def _response_data(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("ok") is False:
        raise RuntimeError(f"Policy server returned an error: {response.get('error', response)}")
    data = response.get("data", response)
    if "actions" not in data:
        raise KeyError(
            f"Key 'actions' not found in policy response. "
                f"data_keys={list(data.keys()) if isinstance(data, dict) else type(data)}, "
                f"response_keys={list(response.keys())}"
        )
    return data
