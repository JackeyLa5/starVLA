#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=${Framework_name:-QwenPI}
freeze_module_list=${freeze_module_list:-}
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3.5-0.8B}
config_yaml=${config_yaml:-./examples/realRobots/Galbot/train_files/starvla_cotrain_galbot.yaml}
galbot_data_root=${galbot_data_root:-/home1/jiajunjie/dataset}
data_mix=${data_mix:-grasp_test}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-galbot_grasp_test}
pretrained_ckpt=${pretrained_ckpt:-}
# === End of environment variable configuration ===
###########################################################################################

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

pretrained_args=()
if [[ -n "${pretrained_ckpt}" ]]; then
  pretrained_args=(--trainer.pretrained_checkpoint "${pretrained_ckpt}")
fi

freeze_args=()
if [[ -n "${freeze_module_list}" ]]; then
  freeze_args=(--trainer.freeze_modules "${freeze_module_list}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.state_dim 21 \
  --framework.action_model.action_dim 21 \
  --framework.action_model.action_horizon 16 \
  "${pretrained_args[@]}" \
  --datasets.vla_data.data_root_dir ${galbot_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.lerobot_version v2.1 \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend torchvision_av \
  "${freeze_args[@]}" \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Galbot \
  --wandb_entity jiajunjie-jack
