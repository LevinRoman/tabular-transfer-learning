#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("188" "1596" "4541" "40664" "40685" "40687" "40975" "41166" "41169" "42734")
declare -a ModelArray=("ft_transformer" "resnet" "mlp" "tab_transformer" "saint")
#4541 is 3 class
#42734 has Nans
#41166?
# Iterate the string array using for loop

#copy the config
for data in ${StringArray[@]}; do
  for model in ${ModelArray[@]}; do
    echo "Launching ${data} ${model}"
    #/bin/bash
    sbatch --qos=very_high --account=tomg --partition=dpart --time=24:00:00 --exclude=cml15,cml11,cml04,cml12 --gres=gpu:1 /cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/tabular-transfer-learning/RTDL/launch/default_experiments/multiclass/saint_data_multiclass_transfer.sh $data $model
  done
done
#"helena" "aloi"

export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA VISIBLE DEVICES SET TO ${CUDA_VISIBLE_DEVICES} FOR 2 GPUS"

for model in ${ModelArray[@]}; do
  echo "Launching covtype ${model}"
  sbatch --qos=very_high --account=tomg --partition=dpart --time=24:00:00 --exclude=cml15,cml11,cml04,cml12 --gres=gpu:1 /cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/tabular-transfer-learning/RTDL/launch/default_experiments/multiclass/ft_data_multiclass_transfer.sh covtype $model
  echo "Launching jannis ${model}"
  sbatch --qos=very_high --account=tomg --partition=dpart --time=24:00:00 --exclude=cml15,cml11,cml04,cml12 --gres=gpu:1 /cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/tabular-transfer-learning/RTDL/launch/default_experiments/multiclass/ft_data_multiclass_transfer.sh jannis $model
  echo "Launching helena ${model}"
  sbatch --qos=very_high --account=tomg --partition=dpart --time=24:00:00 --exclude=cml15,cml11,cml04,cml12 --gres=gpu:2 /cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/tabular-transfer-learning/RTDL/launch/default_experiments/multiclass/ft_data_multiclass_transfer.sh helena $model
  echo "Launching aloi ${model}"
  sbatch --qos=very_high --account=tomg --partition=dpart --time=24:00:00 --exclude=cml15,cml11,cml04,cml12 --gres=gpu:2 /cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/tabular-transfer-learning/RTDL/launch/default_experiments/multiclass/ft_data_multiclass_transfer.sh aloi $model
done

echo "LAUNCHED!"