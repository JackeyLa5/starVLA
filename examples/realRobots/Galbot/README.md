# Galbot Training And Deployment

This directory contains the Galbot-specific StarVLA training setup and the
policy-server adapter used for deployment or simulator evaluation.

## Training

The main training entrypoint is:

```bash
bash examples/realRobots/Galbot/train_files/run_galbot_train.sh
```

Important defaults can be overridden with environment variables:

- `Framework_name`: model framework, default `QwenPI`.
- `base_vlm`: Qwen-VL checkpoint path.
- `config_yaml`: training YAML, default
  `examples/realRobots/Galbot/train_files/starvla_cotrain_galbot.yaml`.
- `galbot_data_root`: LeRobot dataset root.
- `data_mix`: dataset mixture name from
  `examples/realRobots/Galbot/train_files/data_registry/data_config.py`.
- `run_root_dir` and `run_id`: checkpoint output location.
- `pretrained_ckpt`: optional StarVLA checkpoint to resume or finetune from.

By default, training writes checkpoints under:

```text
playground/Checkpoints/galbot_grasp_test
```

The default Galbot policy uses one RGB view:

```text
front_head
```

The state/action layout is 21-dimensional:

```text
legs(5) + left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
```

Head joints are not part of the policy action. They are controlled by the
environment reset or hold helpers.

## Data

`train_files/data_registry/data_config.py` registers the Galbot G1 embodiment,
camera keys, state/action keys, and named dataset mixtures. The default
`grasp_test` mixture uses continuous gripper values without special gripper
post-processing. For experiments that convert raw gripper trajectories into
binary open/close trend labels, use the `grasp_test_gripper_trend` mixture.

The training YAML enables:

- LeRobot `v2.1` input.
- Absolute joint-position action labels.
- 21-D state/action training.
- `obs_image_size: [320, 240]` (`[width, height]`).
- `torchvision_av` video loading.

## Deployment Server

Start the StarVLA policy server with:

```bash
CKPT=/path/to/pytorch_model.pt \
PORT=6694 \
bash examples/realRobots/Galbot/eval_files/run_policy_server.sh
```

The policy server defaults to the final checkpoint produced by the training
script:

```text
playground/Checkpoints/galbot_grasp_test/final_model/pytorch_model.pt
```

Use `CKPT=/path/to/pytorch_model.pt` to evaluate a different checkpoint.

Smoke-test the running server with a fake Galbot client:

```bash
python examples/realRobots/Galbot/eval_files/fake_galbot_client.py \
  --host 127.0.0.1 \
  --port 6694
```

The fake client sends one synthetic `640x360` head-camera RGB image, a zero
21-D Galbot state, and a text instruction. By default it sends 300 sequential
policy requests and prints each returned action shape and first values.

`deployment/model_server/policy_wrapper.py` supports the Galbot deployment
path by returning unnormalized action chunks.

`examples/realRobots/Galbot/eval_files/model2galbot_interface.py` is the Galbot-side
client adapter. It prepares StarVLA websocket requests with the head camera
image by default, keeps RGB inputs in their original resolution for the policy
server to resize consistently with training, and caches action chunks for
step-wise execution.
