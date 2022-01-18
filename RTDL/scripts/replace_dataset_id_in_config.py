from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser(description='Replace dataset id in config')
parser.add_argument('--old_id', type=str, help='old dataset id', default = '1483')
parser.add_argument('--new_id', type = str, help = 'new dataset id')
parser.add_argument('--model', type = str, default = 'ft_transformer', help = 'model to adjust config for')
parser.add_argument('--task', default = 'multiclass_transfer', type = str, help = 'task (e.g. multiclass_transfer, binclass_transfer, sanity_check_transfer)')
parser.add_argument('--seeds', default=['0'], type=str, nargs='+')
parser.add_argument('--experiments', default=['default'], type=str, nargs='+')
parser.add_argument('--transfer_setups', default=['head_fine_tune', 'mlp_head_fine_tune', 'full_fine_tune', 'full_mlp_head_fine_tune'], type=str, nargs='+')
parser.add_argument('--no_transfer_setups', default=['original_model'], type=str, nargs='+')
parser.add_argument('--data_fracs', default=['0.1', '0.01', '0.005'], type=str, nargs='+')
parser.add_argument('--add_apostrophe', action='store_true')
args = parser.parse_args()

if args.new_id is None:
    raise ValueError('New id is not specified')

os.makedirs('output/{}/{}'.format(args.new_id, args.model), exist_ok = True)

assert not os.path.exists('output/{}/{}/{}'.format(args.new_id, args.model, args.task)), 'Model/task directory exists!'

os.system("cp -r output/{}/{}/{} output/{}/{}/{}".format(args.old_id, args.model, args.task, args.new_id, args.model, args.task))


if args.add_apostrophe:
    replace_id = "'{}'".format(args.new_id)
else:
    replace_id = args.new_id

if args.task != 'binclass_transfer':
    pretrain = Path('output/{}/{}/{}/pretrain/default/0.toml'.format(args.new_id, args.model, args.task))
    pretrain.write_text(pretrain.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
    if args.add_apostrophe:
        pretrain.write_text(pretrain.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))


    #No transfer
    for seed in args.seeds:
        for experiment in args.experiments:
            for no_transfer_setup in args.no_transfer_setups:
                for data_frac in args.data_fracs:
                    no_transfer = Path('output/{}/{}/{}/downstream/data_frac_{}/no_transfer/{}/{}/{}.toml'.format(args.new_id,
                                                                                                                       args.model,
                                                                                                                       args.task,
                                                                                                                       data_frac,
                                                                                                                       no_transfer_setup,
                                                                                                                       experiment,
                                                                                                                       seed))
                    no_transfer.write_text(
                        no_transfer.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
                    if args.add_apostrophe:
                        no_transfer.write_text(no_transfer.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))

    #Transfer
    for seed in args.seeds:
        for experiment in args.experiments:
            for transfer_setup in args.transfer_setups:
                for data_frac in args.data_fracs:
                    transfer = Path('output/{}/{}/{}/downstream/data_frac_{}/transfer/{}/{}/{}.toml'.format(args.new_id,
                                                                                                                       args.model,
                                                                                                                       args.task,
                                                                                                                       data_frac,
                                                                                                                       transfer_setup,
                                                                                                                       experiment,
                                                                                                                       seed))
                    transfer.write_text(
                        transfer.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
                    if args.add_apostrophe:
                        transfer.write_text(transfer.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))
else:
    for binary_experiment_num in range(5):
        pretrain = Path('output/{}/{}/{}/pretrain/default/0_{}.toml'.format(args.new_id, args.model, args.task, binary_experiment_num))
        pretrain.write_text(pretrain.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
        if args.add_apostrophe:
            pretrain.write_text(pretrain.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))

        # No transfer
        for seed in args.seeds:
            for experiment in args.experiments:
                for no_transfer_setup in args.no_transfer_setups:
                    for data_frac in args.data_fracs:
                        no_transfer = Path(
                            'output/{}/{}/{}/downstream/data_frac_{}/no_transfer/{}/{}/{}_{}.toml'.format(args.new_id,
                                                                                                       args.model,
                                                                                                       args.task,
                                                                                                       data_frac,
                                                                                                       no_transfer_setup,
                                                                                                       experiment,
                                                                                                       seed,
                                                                                                          binary_experiment_num))
                        no_transfer.write_text(
                            no_transfer.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
                        if args.add_apostrophe:
                            no_transfer.write_text(
                                no_transfer.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))

        # Transfer
        for seed in args.seeds:
            for experiment in args.experiments:
                for transfer_setup in args.transfer_setups:
                    for data_frac in args.data_fracs:
                        transfer = Path('output/{}/{}/{}/downstream/data_frac_{}/transfer/{}/{}/{}_{}.toml'.format(args.new_id,
                                                                                                                args.model,
                                                                                                                args.task,
                                                                                                                data_frac,
                                                                                                                transfer_setup,
                                                                                                                experiment,
                                                                                                                seed,
                                                                                                                   binary_experiment_num))
                        transfer.write_text(
                            transfer.read_text().replace('{}'.format(args.old_id), '{}'.format(replace_id)))
                        if args.add_apostrophe:
                            transfer.write_text(
                                transfer.read_text().replace("/'{}'/".format(args.new_id), "/{}/".format(args.new_id)))



print('Done!')
