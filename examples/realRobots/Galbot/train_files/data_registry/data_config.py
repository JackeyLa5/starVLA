"""Galbot benchmark data config, embodiment tags, and mixtures."""

from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class GalbotGripperTrendTransform(ModalityTransform):
    """Convert raw gripper joint targets into open/close trend labels."""

    jitter_epsilon: float = 0.002
    trend_epsilon: float = 0.01
    debounce_steps: int = 2
    initial_open: bool = True
    gripper_pairs: list[tuple[str, str]] = [
        ("action.left_gripper", "state.left_gripper"),
        ("action.right_gripper", "state.right_gripper"),
    ]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for action_key, state_key in self.gripper_pairs:
            if action_key not in data:
                continue
            action = np.asarray(data[action_key], dtype=np.float32)
            initial = data.get(state_key)
            data[action_key] = self._trend_labels(action, initial).astype(action.dtype)
        return data

    def _trend_labels(self, action: np.ndarray, initial: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return values.reshape(action.shape)

        if initial is None:
            previous_value = float(values[0])
        else:
            previous_value = float(np.asarray(initial, dtype=np.float32).reshape(-1)[0])

        labels = np.ones_like(values, dtype=np.float32) if self.initial_open else np.zeros_like(values, dtype=np.float32)
        is_open = bool(self.initial_open)
        trend_sum = 0.0
        trend_direction = 0
        trend_start_idx = 0
        consecutive_trend_steps = 0

        for idx, value in enumerate(values):
            delta = float(value) - previous_value

            if abs(delta) <= self.jitter_epsilon:
                direction = 0
                effective_delta = 0.0
            else:
                direction = 1 if delta > 0.0 else -1
                effective_delta = delta

            if direction != 0:
                if trend_direction in (0, direction):
                    trend_sum += effective_delta
                    consecutive_trend_steps += 1
                else:
                    trend_sum = effective_delta
                    trend_start_idx = idx
                    consecutive_trend_steps = 1
                trend_direction = direction
            else:
                trend_sum = 0.0
                trend_direction = 0
                trend_start_idx = idx + 1
                consecutive_trend_steps = 0

            if (
                trend_direction == -1
                and trend_sum <= -self.trend_epsilon
                and consecutive_trend_steps >= self.debounce_steps
            ):
                is_open = False
                labels[trend_start_idx : idx + 1] = 0.0
                trend_sum = 0.0
                trend_direction = 0
                trend_start_idx = idx + 1
                consecutive_trend_steps = 0
            elif (
                trend_direction == 1
                and trend_sum >= self.trend_epsilon
                and consecutive_trend_steps >= self.debounce_steps
            ):
                is_open = True
                labels[trend_start_idx : idx + 1] = 1.0
                trend_sum = 0.0
                trend_direction = 0
                trend_start_idx = idx + 1
                consecutive_trend_steps = 0

            labels[idx] = 1.0 if is_open else 0.0
            previous_value = float(value)

        return labels.reshape(action.shape)


class GalbotG1DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front_head_camera_left_color",
    ]
    state_keys = [
        "state.leg",
        "state.left_arm",
        "state.left_gripper",
        "state.right_arm",
        "state.right_gripper",
    ]
    action_keys = [
        "action.leg",
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    action_key_dims = {
        "action.leg": 5,
        "action.left_arm": 7,
        "action.left_gripper": 1,
        "action.right_arm": 7,
        "action.right_gripper": 1,
    }
    state_key_dims = {
        "state.leg": 5,
        "state.left_arm": 7,
        "state.left_gripper": 1,
        "state.right_arm": 7,
        "state.right_gripper": 1,
    }
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
            ]
        )


class GalbotG1GripperTrendDataConfig(GalbotG1DataConfig):
    """Galbot G1 config that converts gripper actions to open/close labels."""

    def transform(self):
        action_norm_modes = {key: "min_max" for key in self.action_keys}
        action_norm_modes["action.left_gripper"] = "binary"
        action_norm_modes["action.right_gripper"] = "binary"

        return ComposedModalityTransform(
            transforms=[
                GalbotGripperTrendTransform(
                    apply_to=[
                        "action.left_gripper",
                        "action.right_gripper",
                    ],
                ),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_norm_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "galbot_g1": GalbotG1DataConfig(),
    "galbot_g1_gripper_trend": GalbotG1GripperTrendDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "grasp_test": [
        ("grasp_test_data", 1.0, "galbot_g1"),
    ],
    "grasp_test_gripper_trend": [
        ("grasp_test_data", 1.0, "galbot_g1_gripper_trend"),
    ],
}
