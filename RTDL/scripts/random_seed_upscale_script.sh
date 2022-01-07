#126
#No transfer downstream
python -c "
for seed in range(1,5):
    open(f'output/126/ft_transformer/binclass_transfer/downstream/no_transfer/{seed}.toml', 'w').write(
        open('output/126/ft_transformer/binclass_transfer/downstream/no_transfer/0.toml').read().replace('seed = 0', f'seed = {seed}')
    )
"

for seed in {1..4}
do
    python bin/ft_transformer.py output/126/ft_transformer/binclass_transfer/downstream/no_transfer/${seed}.toml
done

#Transfer: full fine tune downstream
python -c "
for seed in range(1,5):
    open(f'output/126/ft_transformer/binclass_transfer/downstream/transfer/full_fine_tune_{seed}.toml', 'w').write(
        open('output/126/ft_transformer/binclass_transfer/downstream/transfer/full_fine_tune.toml').read().replace('seed = 0', f'seed = {seed}')
    )
"

for seed in {1..4}
do
    python bin/ft_transformer.py output/126/ft_transformer/binclass_transfer/downstream/transfer/full_fine_tune_${seed}.toml
done

no_t = [0.745489443378119, 0.7488997862441846, 0.7425028470201189, 0.7533888819798532, 0.7577370636296112]
t = [0.7554702495201535, 0.7605934867345656, 0.7576869543211439, 0.7714214649919164, 0.7720970537261699]


