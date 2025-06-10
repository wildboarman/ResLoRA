#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=AD-VTAB
CONFIG=$1
GPUS=1
CKPT=$2
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5

mkdir -p logs
mkdir -p csvs

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

low_rank_dim=8
SEED=0
alpha=10

for LR in 0.001
do
    for GM in param_req_0.1 param_req_0.2 param_req_0.3 param_req_0.4 param_req_0.5 param_req_0.6 param_req_0.8 param_req_1.0 
    do
    for DATASET in cifar caltech101 dtd oxford_flowers102 svhn sun397 oxford_iiit_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
      do
          exp_name=vtab_mae_${GM}_reslora
          export MASTER_PORT=$((12000 + $RANDOM % 20000))
          python train_reslora.py --data-path=/data/vtab-1k/${DATASET} --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_spt --resume=checkpoints/mae_pretrain_vit_base.pth --output_dir=./saves/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY} --batch-size=64 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} --sensitivity_path=sensitivity_reslora_mae_a${alpha}/${DATASET}/${GM}.pth --exp_name=${currenttime}-${exp_name} --seed=0 --test --block=BlockSPTParallel  --structured_type=lora --structured_vector --freeze_stage | tee -a logs/${currenttime}-${exp_name}.log
      done
  done
done