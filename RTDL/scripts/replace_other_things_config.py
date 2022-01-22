from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser(description='Replace dataset id in config')
parser.add_argument('--data_id', type = str, help = 'new dataset id')
parser.add_argument('--model', type = str, default = 'ft_transformer', help = 'model to adjust config for')
parser.add_argument('--task', default = 'multiclass_transfer', type = str, help = 'task (e.g. multiclass_transfer, binclass_transfer, sanity_check_transfer)')
parser.add_argument('--seeds', default=['0'], type=str, nargs='+')
parser.add_argument('--experiments', default=['default'], type=str, nargs='+')
parser.add_argument('--transfer_setups', default=['head_fine_tune', 'mlp_head_fine_tune', 'full_fine_tune', 'full_mlp_head_fine_tune'], type=str, nargs='+')
parser.add_argument('--no_transfer_setups', default=['original_model'], type=str, nargs='+')
parser.add_argument('--data_fracs', default=['250', '50', '10', '5', '2'], type=str, nargs='+')
parser.add_argument('--old_string', type = str, help = 'new dataset id')
parser.add_argument('--new_string', type = str, help = 'new dataset id')
args = parser.parse_args()

if args.data_id is None:
    raise ValueError('Data id is not specified')
if args.old_string is None:
    raise ValueError('Old string is not specified')
if args.new_string is None:
    raise ValueError('New string is not specified')


assert os.path.exists('output/{}/{}/{}'.format(args.data_id, args.model, args.task)), 'Model/task directory does not exist!'

pretrain = Path('output/{}/{}/{}/pretrain/default/0.toml'.format(args.data_id, args.model, args.task))
pretrain.write_text(pretrain.read_text().replace('{}'.format(args.old_string), '{}'.format(args.new_string)))



#No transfer
for seed in args.seeds:
    for experiment in args.experiments:
        for no_transfer_setup in args.no_transfer_setups:
            for data_frac in args.data_fracs:
                no_transfer = Path('output/{}/{}/{}/downstream/data_frac_{}/no_transfer/{}/{}/{}.toml'.format(args.data_id,
                                                                                                                   args.model,
                                                                                                                   args.task,
                                                                                                                   data_frac,
                                                                                                                   no_transfer_setup,
                                                                                                                   experiment,
                                                                                                                   seed))
                no_transfer.write_text(
                    no_transfer.read_text().replace('{}'.format(args.old_string), '{}'.format(args.new_string)))

#Transfer
for seed in args.seeds:
    for experiment in args.experiments:
        for transfer_setup in args.transfer_setups:
            for data_frac in args.data_fracs:
                transfer = Path('output/{}/{}/{}/downstream/data_frac_{}/transfer/{}/{}/{}.toml'.format(args.data_id,
                                                                                                                   args.model,
                                                                                                                   args.task,
                                                                                                                   data_frac,
                                                                                                                   transfer_setup,
                                                                                                                   experiment,
                                                                                                                   seed))
                transfer.write_text(
                    transfer.read_text().replace('{}'.format(args.old_string), '{}'.format(args.new_string)))




print('Done!')