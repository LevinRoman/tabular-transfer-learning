#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("1596" "4541" "40664" "42571" "1111" "1017") #also 188, 1596
# [1596, 44, 54], [4541, 9, 49], [40664, 21, 21], [42571, 72, 130]]
#
#
# From binary:[[1111, 5, 230], [1017, 64, 279]
# Iterate the string array using for loop
for data in ${StringArray[@]}; do

    #copy the config
    python scripts/replace_dataset_id_in_config.py --new_id $data --task binclass_transfer --old_id 4541

    for bin_experiment in {0..5}
    do
        #Pretrain
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/pretrain/default/0_${bin_experiment}.toml -f

        ##################################
        #Data frac 0.1
        #Downstream no transfer
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.1/no_transfer/original_model/default/0_${bin_experiment}.toml -f

        #Downstream transfer
        #head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.1/transfer/head_fine_tune/default/0_${bin_experiment}.toml -f
        #mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.1/transfer/mlp_head_fine_tune/default/0_${bin_experiment}.toml -f
        #full
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.1/transfer/full_fine_tune/default/0_${bin_experiment}.toml -f
        #full mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.1/transfer/full_mlp_head_fine_tune/default/0_${bin_experiment}.toml -f


        ##################################
        #Data frac 0.01
        #Downstream no transfer
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.01/no_transfer/original_model/default/0_${bin_experiment}.toml -f

        #Downstream transfer
        #head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.01/transfer/head_fine_tune/default/0_${bin_experiment}.toml -f
        #mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.01/transfer/mlp_head_fine_tune/default/0_${bin_experiment}.toml -f
        #full
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.01/transfer/full_fine_tune/default/0_${bin_experiment}.toml -f
        #full mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.01/transfer/full_mlp_head_fine_tune/default/0_${bin_experiment}.toml -f



        #################################
        #Data frac 0.005
        #Downstream no transfer
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.005/no_transfer/original_model/default/0_${bin_experiment}.toml -f

        #Downstream transfer
        #head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.005/transfer/head_fine_tune/default/0_${bin_experiment}.toml -f
        #mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.005/transfer/mlp_head_fine_tune/default/0_${bin_experiment}.toml -f
        #full
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.005/transfer/full_fine_tune/default/0_${bin_experiment}.toml -f
        #full mlp head
        python bin/ft_transformer.py output/${data}/ft_transformer/binclass_transfer/downstream/data_frac_0.005/transfer/full_mlp_head_fine_tune/default/0_${bin_experiment}.toml -f
    done

    #tokenizer
    #python bin/ft_transformer.py output/${data}}/ft_transformer/binclass_transfer/downstream/transfer/tokenizer_fine_tune.toml -f
done