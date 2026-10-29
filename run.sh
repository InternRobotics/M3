#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || ${1:-} == --help ]]; then
  echo "Usage: bash run.sh SCENE_PATH MATCHING_CHECKPOINT [pi3|pi3x] [extra arguments]"
  echo "Example: bash run.sh data/my_scene checkpoints/pi3_matching.pt pi3 --calib data/my_scene/intr.yaml"
  exit 0
fi
scene=$1
checkpoint=$2
family=${3:-pi3}
shift 2
if [[ $# -gt 0 ]]; then shift; fi
[[ "$family" == pi3 || "$family" == pi3x ]] || { echo 'Family must be pi3 or pi3x' >&2; exit 2; }
exec python -u run_system.py artdeco3d \
  -s "$scene" -i images -d selfCaptured \
  --model_conf "config/yaml/pi3/${family}match.yaml" --model_weight "$checkpoint" \
  --device_frontend cuda:0 --device_backend cuda:0 --device_mapper cuda:0 --device_shared cpu \
  --config config/base.yaml --viewer_mode none \
  --downsampling 1.0 --test_hold 8 --use_all_frames \
  --num_key_iterations 20 --num_common_iterations 10 \
  --save_at_finetune_iteration 10000 --local_feat_dim 16 --global_feat_dim 16 \
  --visible_threshold 0 --gs_add_ratio 0.3 --num_ref 4 --max_queue_size 8 \
  --covariance_filter --accurate_loop_closure "$@"
