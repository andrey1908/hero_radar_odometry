"""
    PyTorch model for the HERO (Hybrid-Estimate Radar Odometry) network
    Authors: Keenan Burnett, David Yoon
"""
import torch
import numpy as np
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_ref_matcher import SoftmaxRefMatcher
from networks.steam_solver import SteamSolver
from utils.utils import convert_to_weight_matrix, convert_to_radar_frame, mask_intensity_filter
from time import time

class HERO(torch.nn.Module):
    """
        This model performs unsupervised radar odometry using a sliding window optimization with a window
        size between 2 (regular frame-to-frame odometry) and 4. A python wrapper around the STEAM library is used
        to optimize for the best set of transformations over the sliding window.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.window_size = config['window_size']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxRefMatcher(config)
        self.solver = SteamSolver(config)
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.patch_mean_thres = config['steam']['patch_mean_thres']
        self.log_det_thres_flag = config['steam']['log_det_thres_flag']
        self.log_det_thres_val = config['steam']['log_det_thres_val']
        self.log_det_topk = config['steam']['log_det_topk']
        self.no_throw = False
        self.time_used = dict()
        self.time_used['all'] = list()
        self.time_used['feature_map_extraction'] = list()
        self.time_used['keypoint_extraction'] = list()
        self.time_used['keypoint_matching'] = list()
        self.time_used['optimization'] = list()
        self.times = list()

    def forward(self, batch):
        torch.cuda.synchronize()
        time_all = time()

        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)
        timestamps = batch['timestamps']
        t_ref = batch['t_ref']

        BW = data.shape[0]
        B = BW / self.config['window_size']

        torch.cuda.synchronize()
        time_feature_map_extraction = time()
        detector_scores, weight_scores, desc = self.unet(data)
        torch.cuda.synchronize()
        time_feature_map_extraction = time() - time_feature_map_extraction

        # binary mask to remove keypoints from 'empty' regions of the input radar scan
        keypoint_masks = mask_intensity_filter(mask, self.patch_size, self.patch_mean_thres)

        torch.cuda.synchronize()
        time_keypoint_extraction = time()
        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc, keypoint_masks)
        torch.cuda.synchronize()
        time_keypoint_extraction = time() - time_keypoint_extraction

        match_weights_mat = list()
        for i in range(BW):
            tgt_i = i - (i // self.window_size + 1) if i % self.window_size != 0 else None
            weights_mat1, weights_d = convert_to_weight_matrix(keypoint_scores[i][:].T, tgt_i, self.T_aug if tgt_i else None)
            if tgt_i is not None:
                match_weights_mat.append(weights_mat1)
            if self.log_det_thres_flag:
                ids = torch.nonzero(torch.sum(weights_d[:, 0:2], dim=1) > self.log_det_thres_val,
                                    as_tuple=False).squeeze().detach().cpu()
                if ids.squeeze().nelement() <= self.log_det_topk:
                    print('Warning: Log det threshold output less than specified top k.')
                    _, ids = torch.topk(torch.sum(weights_d[:, 0:2], dim=1), self.log_det_topk, largest=True)
                    ids = ids.squeeze().detach().cpu()
                keypoint_coords[i] = keypoint_coords[i][ids, :]
                keypoint_scores[i] = keypoint_scores[i][:, ids]
                keypoint_desc[i] = keypoint_desc[i][:, ids]

        torch.cuda.synchronize()
        time_keypoint_matching = time()
        pseudo_coords, match_weights, tgt_ids, src_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, desc, keypoint_coords)
        torch.cuda.synchronize()
        time_keypoint_matching = time() - time_keypoint_matching

        keypoint_coords_all = keypoint_coords
        keypoint_coords = list()
        for i in range(BW):
            if i not in tgt_ids:
                continue
            keypoint_coords.append(keypoint_coords_all[i])

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.config)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.config)

        # rotate back if augmented
        if 'T_aug' in batch:
            T_aug = torch.stack(batch['T_aug'], dim=0).to(self.gpuid)
            for i in range(len(keypoint_coords_xy)):
                keypoint_coords_xy[i] = torch.matmul(keypoint_coords_xy[i], T_aug[i, :2, :2].transpose(0, 1))
            self.solver.T_aug = batch['T_aug']

        if self.config['flip_y']:
            for i in range(B * (self.config['window_size'] - 1)):
                pseudo_coords_xy[i][:, 1] *= -1.0
                keypoint_coords_xy[i][:, 1] *= -1.0

        time_tgt = torch.index_select(timestamps, 0, tgt_ids.cpu())
        time_src = torch.index_select(timestamps, 0, src_ids.cpu())
        t_ref_tgt = torch.index_select(t_ref, 0, tgt_ids.cpu())
        t_ref_src = torch.index_select(t_ref, 0, src_ids.cpu())
        try:
            torch.cuda.synchronize()
            time_optimization = time()
            R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights_mat,
                                                                  time_tgt, time_src, t_ref_tgt, t_ref_src)
            torch.cuda.synchronize()
            time_optimization = time() - time_optimization
            exception = None
        except Exception as e:
            if not self.no_throw:
                raise
            R_tgt_src_pred = None
            t_tgt_src_pred = None
            exception = e

        torch.cuda.synchronize()
        time_all = time() - time_all

        if exception is None:
            self.time_used['all'].append(time_all)
            self.time_used['feature_map_extraction'].append(time_feature_map_extraction)
            self.time_used['keypoint_extraction'].append(time_keypoint_extraction)
            self.time_used['keypoint_matching'].append(time_keypoint_matching)
            self.time_used['optimization'].append(time_optimization)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores,
                'tgt': keypoint_coords_xy, 'src': pseudo_coords_xy, 'match_weights': match_weights,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords,
                'tgt_ids': tgt_ids, 'src_ids': src_ids, 'keypoint_coords_all': keypoint_coords_all,
                'keypoint_weights_all': keypoint_scores, 'exception': exception}
