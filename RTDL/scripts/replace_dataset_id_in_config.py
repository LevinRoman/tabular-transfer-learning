from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser(description='Replace dataset id in config')
parser.add_argument('--old_id', type=int, help='old dataset id', default = 1483)
parser.add_argument('--new_id', type = int, help = 'new dataset id')
parser.add_argument('--model', type = str, help = 'model to adjust config for')
parser.add_argument('--task', type = str, help = 'task (e.g. multiclass_transfer, binclass_transfer, sanity_check_transfer)')
args = parser.parse_args()

os.makedirs('output/{}/{}'.format(args.new_id, args.model), exist_ok = True)

assert not os.path.exists('output/{}/{}/{}'.format(args.new_id, args.model, args.task)), 'Model/task directory exists!'

os.system("cp -r output/{}/{}/{} output/{}/{}/{}".format(args.old_id, args.model, args.task, args.new_id, args.model, args.task))

pretrain = Path('output/{}/{}/{}/pretrain/default/0.toml'.format(args.new_id, args.model, args.task))
pretrain.write_text(pretrain.read_text().replace('{}'.format(args.old_id), '{}'.format(args.new_id)))

downstream_transfer_head = Path('output/{}/{}/{}/downstream/transfer/head_fine_tune.toml'.format(args.new_id, args.model, args.task))
downstream_transfer_head.write_text(downstream_transfer_head.read_text().replace('{}'.format(args.old_id), '{}'.format(args.new_id)))

downstream_transfer_full = Path('output/{}/{}/{}/downstream/transfer/full_fine_tune.toml'.format(args.new_id, args.model, args.task))
downstream_transfer_full.write_text(downstream_transfer_full.read_text().replace('{}'.format(args.old_id), '{}'.format(args.new_id)))

downstream_no_transfer = Path('output/{}/{}/{}/downstream/no_transfer/0.toml'.format(args.new_id, args.model, args.task))
downstream_no_transfer.write_text(downstream_no_transfer.read_text().replace('{}'.format(args.old_id), '{}'.format(args.new_id)))

print('Done!')