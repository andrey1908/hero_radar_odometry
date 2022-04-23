import argparse
import json
from time import time
import os
import shutil
import numpy as np
import torch

from datasets.oxford import get_dataloaders
from datasets.boreas import get_dataloaders_boreas
from datasets.radiate import get_dataloaders_radiate
from networks.under_the_radar import UnderTheRadar
from networks.hero import HERO
from utils.utils import get_transform2, get_T_ba, computeKittiMetrics, computeMedianError
from utils.vis import plot_sequences, draw_radar, draw_mask, draw_masked_radar, draw_detector_scores, \
    draw_weights, draw_keypoints, draw_src_tgt_matches

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('-no-vis', '--no-visualization', action='store_true')
    parser.add_argument('-out-fld', '--out-folder', type=str, required=True)
    return parser


def makedirs_for_visualization(out_folder):
    os.makedirs(os.path.join(out_folder, 'radar'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'masked_radar_vis'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'detector_scores'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints_only_masked'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints_all'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints_on_detector_scores'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints_on_detector_scores_only_masked'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'keypoints_on_detector_scores_all'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches_only_masked'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches_all'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches_on_detector_scores'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches_on_detector_scores_only_masked'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'src_tgt_matches_on_detector_scores_all'), exist_ok=True)


def visualize(batchi, batch, out, config, out_folder):
    radar_img = draw_radar(batch, i=1)
    radar_img.save(os.path.join(out_folder, 'radar/radar_{}.png'.format(batchi+1)))

    mask_img = draw_mask(batch, i=1)
    mask_img.save(os.path.join(out_folder, 'mask/mask_{}.png'.format(batchi+1)))

    masked_radar_img = draw_masked_radar(batch, i=1)
    masked_radar_img.save(os.path.join(out_folder, 'masked_radar_vis/masked_radar_vis_{}.png'.format(batchi+1)))

    detector_scores_img = draw_detector_scores(out, i=1)
    detector_scores_img.save(os.path.join(out_folder, 'detector_scores/detector_scores_{}.png'.format(batchi+1)))

    weights_img = draw_weights(out, i=1)
    weights_img.save(os.path.join(out_folder, 'weights/weights_{}.png'.format(batchi+1)))

    keypoints_img = draw_keypoints(batch, out, config, i=1, draw_uncertainty_scale=20)
    keypoints_img.save(os.path.join(out_folder, 'keypoints/keypoints_{}.png'.format(batchi+1)))

    keypoints_only_masked_img = draw_keypoints(batch, out, config, i=1, filtering='mask')
    keypoints_only_masked_img.save(os.path.join(out_folder, 'keypoints_only_masked/keypoints_only_masked_{}.png'.format(batchi+1)))

    keypoints_all_img = draw_keypoints(batch, out, config, i=1, filtering='none')
    keypoints_all_img.save(os.path.join(out_folder, 'keypoints_all/keypoints_all_{}.png'.format(batchi+1)))

    keypoints_on_detector_scores_img = draw_keypoints(batch, out, config, i=1, draw_on='detector_scores', draw_uncertainty_scale=20)
    keypoints_on_detector_scores_img.save(os.path.join(out_folder,
        'keypoints_on_detector_scores/keypoints_on_detector_scores_{}.png'.format(batchi+1)))

    keypoints_on_detector_scores_only_masked_img = draw_keypoints(batch, out, config, i=1, draw_on='detector_scores', filtering='mask')
    keypoints_on_detector_scores_only_masked_img.save(os.path.join(out_folder,
        'keypoints_on_detector_scores_only_masked/keypoints_on_detector_scores_only_masked_{}.png'.format(batchi+1)))

    keypoints_on_detector_scores_all_img = draw_keypoints(batch, out, config, i=1, draw_on='detector_scores', filtering='none')
    keypoints_on_detector_scores_all_img.save(os.path.join(out_folder,
        'keypoints_on_detector_scores_all/keypoints_on_detector_scores_all_{}.png'.format(batchi+1)))

    src_tgt_matches_img = draw_src_tgt_matches(batch, out, config, draw_uncertainty_scale=20)
    src_tgt_matches_img.save(os.path.join(out_folder,
        'src_tgt_matches/src_tgt_matches_{}.png'.format(batchi)))

    src_tgt_matches_only_masked_img = draw_src_tgt_matches(batch, out, config, filtering='mask')
    src_tgt_matches_only_masked_img.save(os.path.join(out_folder,
        'src_tgt_matches_only_masked/src_tgt_matches_only_masked_{}.png'.format(batchi)))

    src_tgt_matches_all_img = draw_src_tgt_matches(batch, out, config, filtering='none')
    src_tgt_matches_all_img.save(os.path.join(out_folder,
        'src_tgt_matches_all/src_tgt_matches_all_{}.png'.format(batchi)))

    src_tgt_matches_on_detector_scores_img = draw_src_tgt_matches(batch, out, config, draw_on='detector_scores', draw_uncertainty_scale=20)
    src_tgt_matches_on_detector_scores_img.save(os.path.join(out_folder,
        'src_tgt_matches_on_detector_scores/src_tgt_matches_on_detector_scores_{}.png'.format(batchi)))

    src_tgt_matches_on_detector_scores_only_masked_img = draw_src_tgt_matches(batch, out, config, draw_on='detector_scores', filtering='mask')
    src_tgt_matches_on_detector_scores_only_masked_img.save(os.path.join(out_folder,
        'src_tgt_matches_on_detector_scores_only_masked/src_tgt_matches_on_detector_scores_only_masked_{}.png'.format(batchi)))

    src_tgt_matches_on_detector_scores_all_img = draw_src_tgt_matches(batch, out, config, draw_on='detector_scores', filtering='none')
    src_tgt_matches_on_detector_scores_all_img.save(os.path.join(out_folder,
        'src_tgt_matches_on_detector_scores_all/src_tgt_matches_on_detector_scores_all_{}.png'.format(batchi)))


if __name__ == '__main__':
    torch.set_num_threads(8)
    parser = build_parser()
    args = parser.parse_args()

    out_folder = args.out_folder
    with_visualization = not args.no_visualization
    os.makedirs(out_folder, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    config_copy = os.path.join(out_folder, os.path.basename(args.config))
    if args.config != config_copy:
        shutil.copy(args.config, config_copy)

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

    seq_names = list()
    time_used_all = list()
    T_gt_all = list()
    T_pred_all = list()
    t_errs = list()
    r_errs = list()
    seq_nums = config['test_split']
    for seq_num in seq_nums:
        time_used = list()
        T_gt = list()
        T_pred = list()
        timestamps = list()

        config['test_split'] = [seq_num]
        if config['dataset'] == 'oxford':
            _, _, test_loader = get_dataloaders(config)
        elif config['dataset'] == 'boreas':
            _, _, test_loader = get_dataloaders_boreas(config)
        elif config['dataset'] == 'radiate':
            _, _, test_loader = get_dataloaders_radiate(config)

        seq_len = test_loader.dataset.seq_lens[0]
        seq_name = test_loader.dataset.sequences[0]
        print('Evaluating sequence {} (len {}): {}'.format(seq_num, seq_len, seq_name))

        if with_visualization:
            out_vis_folder = os.path.join(out_folder, seq_name)
            makedirs_for_visualization(out_vis_folder)

        for batchi, batch in enumerate(test_loader):
            ts = time()

            try:
                with torch.no_grad():
                    out = model(batch)
            except:
                with open(os.path.join(out_folder, 'failed.txt'), 'w') as f:
                    f.write('{}'.format(batchi))
                raise

            if with_visualization and batchi % config['vis_rate'] == 0:
                visualize(batchi, batch, out, config, out_vis_folder)

            if config['model'] == 'UnderTheRadar':
                if 'T_21' in batch:
                    T_gt.append(batch['T_21'][0].numpy().squeeze())
                R_pred = out['R'][0].detach().cpu().numpy().squeeze()
                t_pred = out['t'][0].detach().cpu().numpy().squeeze()
                T_pred.append(get_transform2(R_pred, t_pred))
            elif config['model'] == 'HERO':
                if batchi == len(test_loader) - 1:
                    for w in range(config['window_size'] - 1):
                        if 'T_21' in batch:
                            T_gt.append(batch['T_21'][w].numpy().squeeze())
                        T_pred.append(get_T_ba(out, a=w, b=w+1))
                        timestamps.append(batch['t_ref'][w].numpy().squeeze())
                else:
                    w = 0
                    if 'T_21' in batch:
                        T_gt.append(batch['T_21'][w].numpy().squeeze())
                    T_pred.append(get_T_ba(out, a=w, b=w+1))
                    timestamps.append(batch['t_ref'][w].numpy().squeeze())

            time_used.append(time() - ts)
            if (batchi + 1) % config['print_rate'] == 0:
                print('Eval Batch {} / {}: {:.2}s'.format(batchi, len(test_loader), np.mean(time_used[-config['print_rate']:])))

        time_used_all.extend(time_used)
        if len(T_gt) > 0:
            seq_names.append(seq_name)
            T_gt_all.extend(T_gt)
            T_pred_all.extend(T_pred)
            t_err, r_err = computeKittiMetrics(T_gt, T_pred, [len(T_gt)])
            print('SEQ: {} : {}'.format(seq_num, seq_name))
            print('KITTI t_err: {} %'.format(t_err))
            print('KITTI r_err: {} deg/m'.format(r_err))
            t_errs.append(t_err)
            r_errs.append(r_err)
        fname = os.path.join(out_folder, seq_name + '.png')
        if len(T_gt) > 0:
            plot_sequences(T_gt, T_pred, [len(T_pred)], returnTensor=False, savePDF=True, fnames=[fname])
        else:
            plot_sequences(T_pred, T_pred, [len(T_pred)], returnTensor=False, savePDF=True, fnames=[fname])

    print('time_used: {}'.format(sum(time_used_all) / len(time_used_all)))

    if len(T_gt_all) > 0:
        results = computeMedianError(T_gt_all, T_pred_all)
        print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

        t_err_mean = np.mean(t_errs)
        r_err_mean = np.mean(r_errs)
        print('Average KITTI metrics over all test sequences:')
        print('KITTI t_err: {} %'.format(t_err_mean))
        print('KITTI r_err: {} deg/m'.format(r_err_mean))

        with open(os.path.join(out_folder, 'metrics.txt'), 'w') as f:
            f.write('sequence name: translation error (%) rotation error (deg/m)\n')
            for seq_name, t_err, r_err in zip(seq_names, t_errs, r_errs):
                line = '{}: {} {}\n'.format(seq_name, t_err, r_err)
                f.write(line)
