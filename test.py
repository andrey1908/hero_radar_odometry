import argparse
import json
from time import time
import os
import numpy as np
import torch

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from datasets.radiate import get_dataloaders_radiate
from networks.under_the_radar import UnderTheRadar
from networks.hero import HERO
from utils.utils import get_T_ba
from utils.utils import get_transform2
from utils.vis import plot_sequences

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('-out-fld', '--out-folder', type=str, required=True)
    return parser


if __name__ == '__main__':
    torch.set_num_threads(8)
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    if config['model'] == 'UnderTheRadar':
        model = UnderTheRadar(config).to(config['gpuid'])
    elif config['model'] == 'HERO':
        model = HERO(config).to(config['gpuid'])
        model.solver.sliding_flag = True
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(config['gpuid']))
    failed = False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        print(e)
        failed = True
    if failed:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    T_pred_all = list()
    time_used_all = list()
    seq_nums = config['test_split']
    for seq_num in seq_nums:
        time_used = []
        T_pred = []
        timestamps = []
        config['test_split'] = [seq_num]
        if config['dataset'] == 'oxford':
            _, _, test_loader = get_dataloaders(config)
        elif config['dataset'] == 'boreas':
            _, _, test_loader = get_dataloaders_boreas(config)
        elif config['dataset'] == 'radiate':
            _, _, test_loader = get_dataloaders_radiate(config)

        seq_lens = test_loader.dataset.seq_lens
        print(seq_lens)
        seq_names = test_loader.dataset.sequences
        print('Evaluating sequence: {} : {}'.format(seq_num, seq_names[0]))
        for batchi, batch in enumerate(test_loader):
            ts = time()
            with torch.no_grad():
                out = model(batch)
            if config['model'] == 'UnderTheRadar':
                R_pred_ = out['R'][0].detach().cpu().numpy().squeeze()
                t_pred_ = out['t'][0].detach().cpu().numpy().squeeze()
                T_pred.append(get_transform2(R_pred_, t_pred_))
            elif config['model'] == 'HERO':
                if batchi == len(test_loader) - 1:
                    for w in range(config['window_size'] - 1):
                        T_pred.append(get_T_ba(out, a=w, b=w+1))
                        timestamps.append(batch['t_ref'][w].numpy().squeeze())
                else:
                    w = 0
                    T_pred.append(get_T_ba(out, a=w, b=w+1))
                    timestamps.append(batch['t_ref'][w].numpy().squeeze())
            time_used.append(time() - ts)
            if (batchi + 1) % config['print_rate'] == 0:
                print('Eval Batch {} / {}: {:.2}s'.format(batchi, len(test_loader), np.mean(time_used[-config['print_rate']:])))
        T_pred_all.extend(T_pred)
        time_used_all.extend(time_used)
        fname = os.path.join(args.out_folder, seq_names[0] + '.pdf')
        plot_sequences(T_pred, T_pred, [len(T_pred)], returnTensor=False, savePDF=True, fnames=[fname])

    print('time_used: {}'.format(sum(time_used_all) / len(time_used_all)))
