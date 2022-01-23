#!/bin/bash

data=$1
model=$2
#copy the config
python scripts/replace_dataset_id_in_config.py --new_id $data --old_id 1483 --model $model

#Pretrain
python bin/${model}.py output/${data}/${model}/multiclass_transfer/pretrain/default/0.toml

##################################
#Samples per class: 250
#Downstream no transfer
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/no_transfer/original_model/default/0.toml -f

#Downstream transfer
#head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/head_fine_tune/default/0.toml -f
#$model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/mlp_head_fine_tune/default/0.toml -f
#full
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune/default/0.toml -f
#full $model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/full_mlp_head_fine_tune/default/0.toml -f
#full big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune_big_lr/default/0.toml -f
#full $model head big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_250/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


##################################
#Samples per class: 50
#Downstream no transfer
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/no_transfer/original_model/default/0.toml -f

#Downstream transfer
#head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/head_fine_tune/default/0.toml -f
#$model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/mlp_head_fine_tune/default/0.toml -f
#full
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune/default/0.toml -f
#full $model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/full_mlp_head_fine_tune/default/0.toml -f
#full big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune_big_lr/default/0.toml -f
#full $model head big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_50/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f



#################################
#Samples per class: 10
#Downstream no transfer
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/no_transfer/original_model/default/0.toml -f

#Downstream transfer
#head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/head_fine_tune/default/0.toml -f
#$model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/mlp_head_fine_tune/default/0.toml -f
#full
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune/default/0.toml -f
#full $model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/full_mlp_head_fine_tune/default/0.toml -f
#full big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune_big_lr/default/0.toml -f
#full $model head big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_10/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


#################################
#Samples per class: 5
#Downstream no transfer
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/no_transfer/original_model/default/0.toml -f

#Downstream transfer
#head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/head_fine_tune/default/0.toml -f
#$model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/mlp_head_fine_tune/default/0.toml -f
#full
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune/default/0.toml -f
#full $model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/full_mlp_head_fine_tune/default/0.toml -f
#full big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune_big_lr/default/0.toml -f
#full $model head big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_5/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


#################################
#Samples per class: 2
#Downstream no transfer
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/no_transfer/original_model/default/0.toml -f

#Downstream transfer
#head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/head_fine_tune/default/0.toml -f
#$model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/mlp_head_fine_tune/default/0.toml -f
#full
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune/default/0.toml -f
#full $model head
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/full_mlp_head_fine_tune/default/0.toml -f
#full big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune_big_lr/default/0.toml -f
#full $model head big lr
python bin/${model}.py output/${data}/${model}/multiclass_transfer/downstream/data_frac_2/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


#tokenizer
#python bin/${model}}.py output/${data}}/${model}}/multiclass_transfer/downstream/transfer/tokenizer_fine_tune.toml -f
