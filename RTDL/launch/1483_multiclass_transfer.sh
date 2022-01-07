
#Pretrain
python bin/ft_transformer.py output/1483/ft_transformer/multiclass_transfer/pretrain/default/0.toml -f

#Downstream no transfer
python bin/ft_transformer.py output/1483/ft_transformer/multiclass_transfer/downstream/no_transfer/0.toml -f

#Downstream transfer
#head
python bin/ft_transformer.py output/1483/ft_transformer/multiclass_transfer/downstream/transfer/head_fine_tune.toml -f
#full
python bin/ft_transformer.py output/1483/ft_transformer/multiclass_transfer/downstream/transfer/full_fine_tune.toml -f
#tokenizer
python bin/ft_transformer.py output/1483/ft_transformer/multiclass_transfer/downstream/transfer/tokenizer_fine_tune.toml -f

