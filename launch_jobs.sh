#!/bin/sh
#SBATCH -e slurm_logs/log.err
#SBATCH -o slurm_logs/log.out
#SBATCH -p long
#SBATCH --gres gpu:8
#SBATCH -J vqa_introspect_{$1}
#SBATCH --qos=overcap
WEIGHT=$1
tmux new -s $WEIGHT
CONFIG_YML="configs/vqa/vqa2/pythia_introspect.yml"
srun python tools/run.py \
    --tasks vqa \
    --datasets vqa_introspect \
    --model pythia_introspect \
    --config ${CONFIG_YML} \
    --resume_file data/models/pythia.pth \
    --save_dir ./save_clean_${WEIGHT} \
    --config_override '{"model_attributes":{"pythia":{"losses":[{"type":"multi", "params":[{"params":{}, "type":"logit_bce", "weight":1}, {"params":{}, "type":"sq_loss", "weight":'$WEIGHT'}]}]}}}'
