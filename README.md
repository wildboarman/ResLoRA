# Feature Responsive LoRA: Towards Parameter-Efficient Transfer Learning for Self-Supervised Visual Models

## Install dependency:

```
pip install -r requirements.txt

```
## Data preparation:

```
cd data/vtab-source
python get_vtab1k.py
```
PS: You may have to manually install Sun397. Please refer to [VTAB-1k](https://github.com/google-research/task_adaptation).

## Download pre-trained models:

```
cd checkpoints

# Supervised pre-trained ViT-B/16
wget https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz

# MAE pre-trained ViT-B/16
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# MoCo V3 pre-trained ViT-B/16
wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar
```

## Get Salient Parameters and LoRA-sets with MAE pre-trained ViT-B/16:
```
bash vtab_mae_reslora_sensitivity.sh
```
we have already provided the sets for MAE pre-trained ViT-B/16
## Run ResLoRA with MAE pre-trained ViT-B/16:
```
bash vtab_mae_reslora.sh
```




