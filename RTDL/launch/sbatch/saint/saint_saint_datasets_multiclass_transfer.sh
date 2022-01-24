#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("188" "1596" "4541" "40664" "40685" "40687" "40975" "41166" "41169" "42734") #also 188, 1596
#4541 is 3 class
#42734 has Nans
#41166?
#TAKING: 188, 40687
#40975 -- categorical only, default saint doesn't work
#41166 -- 2 GPUs
#41169 -- 100 class helena
#1483
#jannis (41168)
#Regression: 541 42726 42728 california housing, year
#188, 40687, 40975, 41166, 41169, 1483, jannis, regression:  541 42726 42728 california housing, year
# Iterate the string array using for loop
for data in ${StringArray[@]}; do
    #copy the config
    python scripts/replace_dataset_id_in_config.py --new_id $data --old_id 1483 --model saint --do_destructive_danger

    #Pretrain
    python bin/saint.py output/${data}/saint/multiclass_transfer/pretrain/default/0.toml -f

    ##################################
    #Samples per class: 250
    #Downstream no transfer
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/head_fine_tune/default/0.toml -f
    #saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune/default/0.toml -f
    #full saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/full_mlp_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full saint head big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_250/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


    ##################################
    #Samples per class: 50
    #Downstream no transfer
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/head_fine_tune/default/0.toml -f
    #saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune/default/0.toml -f
    #full saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/full_mlp_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full saint head big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_50/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f



    #################################
    #Samples per class: 10
    #Downstream no transfer
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/head_fine_tune/default/0.toml -f
    #saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune/default/0.toml -f
    #full saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/full_mlp_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full saint head big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_10/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


    #################################
    #Samples per class: 5
    #Downstream no transfer
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/head_fine_tune/default/0.toml -f
    #saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune/default/0.toml -f
    #full saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/full_mlp_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full saint head big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_5/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


    #################################
    #Samples per class: 2
    #Downstream no transfer
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/head_fine_tune/default/0.toml -f
    #saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune/default/0.toml -f
    #full saint head
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/full_mlp_head_fine_tune/default/0.toml -f
    #full big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/full_fine_tune_big_lr/default/0.toml -f
    #full saint head big lr
    python bin/saint.py output/${data}/saint/multiclass_transfer/downstream/data_frac_2/transfer/full_mlp_head_fine_tune_big_lr/default/0.toml -f


    #tokenizer
    #python bin/saint.py output/${data}}/saint/multiclass_transfer/downstream/transfer/tokenizer_fine_tune.toml -f
done