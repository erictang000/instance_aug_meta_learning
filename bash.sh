#!/bin/sh
################# Query Augmentation ####################

######### 1 Shot ########
## No Augmentations Baseline
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_no_aug" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --no_color_jitter --no_random_cropping --wandb True

## CutMix for comparison
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_cutmix" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug cutmix --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## RandomResizedCropping for comparison - need to figure out how to apply on just query set
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_random_crop" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_instacrop_3_3.5_new_scheduler" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_instacrop_3.5_4" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3.5 --max_entropy 4 --wandb True

######### 5 Shot ########
## No Augmentations Baseline
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_no_aug" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --no_color_jitter --no_random_cropping --wandb True

## CutMix for comparison
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_cutmix" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug cutmix --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## RandomResizedCropping for comparison - need to figure out how to apply on just query set
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_random_crop" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

# RandomErase
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_random_erase" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug random_erase --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

# ColorJitter
# python train_aug.py --gpu 2 --save-path "./experiments/CIFAR_5_shot_query_color_jitter" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug color_jitter --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_instacrop_3_3.5" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_instacrop_2.5_3" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 2.5 --max_entropy 3 --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_instacrop_3.5_4" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3.5 --max_entropy 4 --wandb True

######### 10 Shot ########
## No Augmentations Baseline
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_no_aug" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --no_color_jitter --no_random_cropping --wandb True

## CutMix for comparison
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_cutmix" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug cutmix --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## RandomResizedCropping for comparison - need to figure out how to apply on just query set
# python train_aug.py --gpu 1 --save-path "./experiments/CIFAR_10_shot_query_random_crop" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_instacrop_3_3.5" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_instacrop_2.5_3" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 2.5 --max_entropy 3 --wandb True

# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_instacrop_3.5_4" --train-shot 10--val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3.5 --max_entropy 4 --wandb True


################# Query + Support Augmentation ####################
######### 1 Shot ########
## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_support_instacrop_3_3.5" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug instance --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True
## RandomResizedCropping for comparison
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_shot_random_crop" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug random_crop --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

######### 5 Shot ########
# ## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_support_instacrop_3_3.5" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug instance --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

## RandomResizedCropping for comparison
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_shot_random_crop" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug random_crop --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

######### 10 Shot ########
# ## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_support_instacrop_3_3.5" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug instance --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

## RandomResizedCropping for comparison 
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_shot_random_crop" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --support_aug random_crop --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

################# Query + Shot Augmentation ####################
######### 1 Shot ########
## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_shot_instacrop_3_3.5" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug instance --s_du 2 --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

## RandomResizedCropping for comparison - try on shot + query
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_1_shot_query_shot_random_crop" --train-shot 1 --val-shot 1 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug random_crop --s_du 2 --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True

######### 5 Shot ########
# ## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_shot_instacrop_3_3.5" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug instance --s_du 2 --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

## RandomResizedCropping for comparison - try on shot + query
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_5_shot_query_support_random_crop" --train-shot 5 --val-shot 5 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug random_crop --s_du 2 --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True


######### 10 Shot ########
# ## Instance Cropping
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_shot_instacrop_3_3.5" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug instance --s_du 2 --query_aug instance --q_p 1. --Li_config_path ./InstaAug_module/configs/config_crop_supervised_cifar.yaml \
#    --no_color_jitter --no_random_cropping --min_entropy 3. --max_entropy 3.5 --wandb True

## RandomResizedCropping for comparison - try on shot + query
# python train_aug.py --gpu 0 --save-path "./experiments/CIFAR_10_shot_query_support_random_crop" --train-shot 10 --val-shot 10 \
#    --head ProtoNet --network ResNet --dataset CIFAR_FS --shot_aug random_crop --s_du 2 --query_aug random_crop --q_p 1. \
#    --no_color_jitter --no_random_cropping --wandb True