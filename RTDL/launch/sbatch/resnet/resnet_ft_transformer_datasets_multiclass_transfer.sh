#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("helena" "aloi" "covtype" "jannis")

# Iterate the string array using for loop
for data in ${StringArray[@]}; do
    #copy the config
    python scripts/replace_dataset_id_in_config.py --new_id ${data} --add_apostrophe --old_id 1483 --model resnet --do_destructive_danger

    #Pretrain
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/pretrain/default/0.toml -f

    ##################################
    #Samples per class: 250
    #Downstream no transfer
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/head_fine_tune/default/0.toml -f
    #resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/resnet_head_fine_tune/default/0.toml -f
    #full
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune/default/0.toml -f
    #full resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/full_resnet_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full resnet head big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_250/transfer/full_resnet_head_fine_tune_big_lr/default/0.toml -f


    ##################################
    #Samples per class: 50
    #Downstream no transfer
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/head_fine_tune/default/0.toml -f
    #resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/resnet_head_fine_tune/default/0.toml -f
    #full
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune/default/0.toml -f
    #full resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/full_resnet_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full resnet head big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_50/transfer/full_resnet_head_fine_tune_big_lr/default/0.toml -f



    #################################
    #Samples per class: 10
    #Downstream no transfer
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/head_fine_tune/default/0.toml -f
    #resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/resnet_head_fine_tune/default/0.toml -f
    #full
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune/default/0.toml -f
    #full resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/full_resnet_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full resnet head big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_10/transfer/full_resnet_head_fine_tune_big_lr/default/0.toml -f


    #################################
    #Samples per class: 5
    #Downstream no transfer
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/head_fine_tune/default/0.toml -f
    #resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/resnet_head_fine_tune/default/0.toml -f
    #full
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune/default/0.toml -f
    #full resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/full_resnet_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full resnet head big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_5/transfer/full_resnet_head_fine_tune_big_lr/default/0.toml -f


    #################################
    #Samples per class: 2
    #Downstream no transfer
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/head_fine_tune/default/0.toml -f
    #resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/resnet_head_fine_tune/default/0.toml -f
    #full
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune/default/0.toml -f
    #full resnet head
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/full_resnet_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full resnet head big lr
    python bin/resnet.py output/${data}/resnet/multiclass_transfer/downstream/data_frac_2/transfer/full_resnet_head_fine_tune_big_lr/default/0.toml -f


    #tokenizer
    #python bin/resnet.py output/${data}}/resnet/multiclass_transfer/downstream/transfer/tokenizer_fine_tune.toml -f
done