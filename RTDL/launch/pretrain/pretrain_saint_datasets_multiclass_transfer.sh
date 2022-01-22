#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("4541" "40664" "40685" "40687" "40975" "41166" "41169" "42734") # "188" "1596"#also 188, 1596
#4541 has 3 classes only and is not suitable
# Iterate the string array using for loop
for data in ${StringArray[@]}; do
    #copy the config
    python scripts/replace_dataset_id_in_config.py --new_id $data --old_id 1483

    #Pretrain
    python bin/ft_transformer.py output/${data}/ft_transformer/multiclass_transfer/pretrain/default/0.toml -f


done