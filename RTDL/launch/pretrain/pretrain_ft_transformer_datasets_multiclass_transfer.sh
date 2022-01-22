#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("aloi") #"helena" "aloi" "covtype" "jannis"

# Iterate the string array using for loop
for data in ${StringArray[@]}; do
    #copy the config
    python scripts/replace_dataset_id_in_config.py --new_id ${data} --add_apostrophe --old_id 1483

    #Pretrain
    python bin/ft_transformer.py output/${data}/ft_transformer/multiclass_transfer/pretrain/default/0.toml -f

done