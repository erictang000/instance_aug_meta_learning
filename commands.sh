#!/bin/sh
## ProtoNet with CutMix and Rotation task augmentation
# python train_aug.py --gpu 0 --save-path "./experiments/ResNet_ProtoNet_qcm_tlr" --train-shot 5  \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug cutmix --q_p 1. --task_aug Rot90 --t_p 0.25 --wandb True

## ProtoNet with InstanceAug + rotation task aug
# python train_aug.py --gpu 0 --save-path "./experiments/ResNet_ProtoNet_qia_tlr" --train-shot 5  \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --wandb True

## no aug
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_ResNet_ProtoNet_instance_aug_1_shot" --train-shot 1 --val-shot 1  \
#   --head ProtoNet --network ResNet --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 --dataset CIFAR_FS --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --wandb True --train_instance True

# python train_aug.py --gpu 0 --save-path "./experiments/ResNet_ProtoNet_rot_5_shot" --train-shot 5  \
#   --head ProtoNet --network ResNet --task_aug Rot90 --t_p 0.25 --dataset CIFAR_FS --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/ResNet_ProtoNet_qia_tlr" --train-shot 1  \
#   --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --wandb True --train_instance True

# python train_aug.py --gpu 0 --save-path "./experiments/ResNet_ProtoNet_qia_tlr_tiny" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet --dataset miniImageNet --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 --Li_config_path ./InstaAug_module/configs/config_crop_supervised.yaml --train_instance True

# python train_aug.py --gpu 0 --save-path "./experiments/cifar_instance_color_jitter_5_shot_rot90" --train-shot 5 --val-shot 5 \
#   --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 --Li_config_path ./InstaAug_module/configs/config_color_jittering_supervised_cifar.yaml --train_instance True --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/cifar_random_crop_no_jitter_5_shot_rot90" --train-shot 5 --val-shot 5 \
#   --head ProtoNet --network ResNet --dataset CIFAR_FS --task_aug Rot90 --t_p 0.25 --wandb True
################# Query Augmentation ####################
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_q_instance_crop_1_shot" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping

################# Query + Support Augmentation ####################
##### CIFAR_FS 1 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_qs_instance_crop_1_shot" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --support_aug instance --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True

##### CIFAR_FS 1 Shot Jitter #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_qs_instance_color_jitter_1_shot" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --support_aug instance --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_color_jittering_supervised_cifar.yaml --train_instance True \
#   --no_color_jitter  --wandb True

##### CIFAR_FS 5 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_qs_instance_crop_5_shot" --train-shot 5 --val-shot 5 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --support_aug instance --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True

##### CIFAR_FS 5 Shot Jitter #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_qs_instance_color_jitter_5_shot" --train-shot 5 --val-shot 5 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --support_aug instance --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_color_jittering_supervised_cifar.yaml --train_instance True \
#   --no_color_jitter  --wandb True

#### CIFAR_FS 10 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_q_instance_crop_10_shot" --train-shot 10 --val-shot 10 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True

### CIFAR_FS 10 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_q_random_crop_10_shot" --train-shot 10 --val-shot 10 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --task_aug Rot90 --t_p 0.25 \
#   --wandb True

#### CIFAR_FS 15 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_q_instance_crop_15_shot" --train-shot 15 --val-shot 15 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True

##### miniImageNet 1 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/miniImageNet_qs_instance_crop_1_shot" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet \
#   --dataset miniImageNet \
#   --support_aug instance --query_aug instance --q_p 1. --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_mini.yaml --train_instance True \
#   --no_random_cropping --wandb True

##### miniImageNet 1 Shot Jitter #####

# Shot Augmentation
##### CIFAR_FS 5 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_shotaug_instance_crop_5_shot" --train-shot 5 --val-shot 5 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --shot_aug instance --s_du 2 --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True

##### CIFAR_FS 1 Shot Cropping #####
# python train_aug.py --gpu 0 --save-path "./experiments/cifar_shotaug_instance_crop_1_shot" --train-shot 1 --val-shot 1 \
#   --head ProtoNet --network ResNet \
#   --dataset CIFAR_FS \
#   --shot_aug instance --s_du 2 --task_aug Rot90 --t_p 0.25 \
#   --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml --train_instance True \
#   --no_random_cropping --wandb True
