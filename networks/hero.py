"""
    PyTorch model for the HERO (Hybrid-Estimate Radar Odometry) network
    Authors: Keenan Burnett, David Yoon
"""
import torch
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_ref_matcher import SoftmaxRefMatcher
from networks.steam_solver import SteamSolver
from utils.utils import convert_to_radar_frame, mask_intensity_filter
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
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxRefMatcher(config)
        self.solver = SteamSolver(config)
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.patch_mean_thres = config['steam']['patch_mean_thres']
        self.no_throw = False
        self.time_used = dict()
        self.time_used['all'] = list()
        self.time_used['feature_map_extraction'] = list()
        self.time_used['keypoint_extraction'] = list()
        self.time_used['keypoint_matching'] = list()
        self.time_used['optimization'] = list()

    def forward(self, batch):
        time_all = time()

        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)
        timestamps = batch['timestamps']
        t_ref = batch['t_ref']

        time_feature_map_extraction = time()
        detector_scores, weight_scores, desc = self.unet(data)
        time_feature_map_extraction = time() - time_feature_map_extraction

        time_keypoint_extraction = time()
        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)
        time_keypoint_extraction = time() - time_keypoint_extraction

        time_keypoint_matching = time()
        pseudo_coords, match_weights, tgt_ids, src_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, desc, keypoint_coords)
        time_keypoint_matching = time() - time_keypoint_matching

        all_keypoint_coords = keypoint_coords
        keypoint_coords = keypoint_coords[tgt_ids]

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.config)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.config)
        # rotate back if augmented
        if 'T_aug' in batch:
            T_aug = torch.stack(batch['T_aug'], dim=0).to(self.gpuid)
            keypoint_coords_xy = torch.matmul(keypoint_coords_xy, T_aug[:, :2, :2].transpose(1, 2))
            self.solver.T_aug = batch['T_aug']

        if self.config['flip_y']:
            pseudo_coords_xy[:, :, 1] *= -1.0
            keypoint_coords_xy[:, :, 1] *= -1.0

        # binary mask to remove keypoints from 'empty' regions of the input radar scan
        all_keypoint_ints = mask_intensity_filter(mask, self.patch_size, self.patch_mean_thres)
        keypoint_ints = all_keypoint_ints[tgt_ids]

        time_tgt = torch.index_select(timestamps, 0, tgt_ids.cpu())
        time_src = torch.index_select(timestamps, 0, src_ids.cpu())
        t_ref_tgt = torch.index_select(t_ref, 0, tgt_ids.cpu())
        t_ref_src = torch.index_select(t_ref, 0, src_ids.cpu())
        try:
            time_optimization = time()
            R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights,
                                                                keypoint_ints, time_tgt, time_src, t_ref_tgt, t_ref_src)
            time_optimization = time() - time_optimization
            exception = None
        except Exception as e:
            if not self.no_throw:
                raise
            R_tgt_src_pred = None
            t_tgt_src_pred = None
            exception = e

        time_all = time() - time_all

        if exception is None:
            self.time_used['all'].append(time_all)
            self.time_used['feature_map_extraction'].append(time_feature_map_extraction)
            self.time_used['keypoint_extraction'].append(time_keypoint_extraction)
            self.time_used['keypoint_matching'].append(time_keypoint_matching)
            self.time_used['optimization'].append(time_optimization)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'tgt': keypoint_coords_xy,
                'src': pseudo_coords_xy, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords,
                'tgt_ids': tgt_ids, 'src_ids': src_ids, 'all_keypoint_coords': all_keypoint_coords,
                'all_keypoint_ints': all_keypoint_ints, 'all_keypoint_weights': keypoint_scores, 'exception': exception}
