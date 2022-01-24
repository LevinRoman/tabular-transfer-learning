#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("microsoft") #declare -a StringArray=("california_housing" "yahoo" "year" "microsoft") #

#microsoft, yahoo out of memory./lau
# Iterate the string array using for loop
for data in ${StringArray[@]}; do
    #copy the config
    python scripts/replace_dataset_id_in_config.py --old_id 541 --new_id ${data} --add_apostrophe --task regression_transfer

    #Pretrain
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/pretrain/default/0.toml -f

    ##################################
    #Data frac 0.1
    #Downstream no transfer
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.1/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.1/transfer/head_fine_tune/default/0.toml -f
    #mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.1/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.1/transfer/full_fine_tune/default/0.toml -f
    #full mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.1/transfer/full_mlp_head_fine_tune/default/0.toml -f


    ##################################
    #Data frac 0.01
    #Downstream no transfer
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.01/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.01/transfer/head_fine_tune/default/0.toml -f
    #mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.01/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.01/transfer/full_fine_tune/default/0.toml -f
    #full mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.01/transfer/full_mlp_head_fine_tune/default/0.toml -f



    #################################
    #Data frac 0.005
    #Downstream no transfer
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.005/no_transfer/original_model/default/0.toml -f

    #Downstream transfer
    #head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.005/transfer/head_fine_tune/default/0.toml -f
    #mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.005/transfer/mlp_head_fine_tune/default/0.toml -f
    #full
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.005/transfer/full_fine_tune/default/0.toml -f
    #full mlp head
    python bin/ft_transformer.py output/${data}/ft_transformer/regression_transfer/downstream/data_frac_0.005/transfer/full_mlp_head_fine_tune/default/0.toml -f



    #tokenizer
    #python bin/ft_transformer.py output/${data}}/ft_transformer/regression_transfer/downstream/transfer/tokenizer_fine_tune.toml -f
done