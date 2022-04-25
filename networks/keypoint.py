import torch
import torch.nn.functional as F
from utils.utils import normalize_coords

from time import time
import numpy as np

class Keypoint(torch.nn.Module):
    """
        Given a dense map of detector scores and weight scores, this modules computes keypoint locations, and their
        associated scores and descriptors. A spatial softmax is used over a regular grid of "patches" to extract a
        single location, score, and descriptor per patch.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        self.v_coords = v_coords.unsqueeze(0).float()  # (1,H,W)
        self.u_coords = u_coords.unsqueeze(0).float()
        self.v_patches = F.unfold(self.v_coords.view(1, 1, self.width, self.width), kernel_size=self.patch_size,
                             stride=self.patch_size).to(self.gpuid)  # (1,patch_elems,num_patches)
        self.u_patches = F.unfold(self.u_coords.view(1, 1, self.width, self.width), kernel_size=self.patch_size,
                             stride=self.patch_size).to(self.gpuid)
        self.times = list()

    def forward(self, detector_scores, weight_scores, descriptors, keypoint_masks):
        """ A spatial softmax is performed for each grid cell over the detector_scores tensor to obtain 2D
            keypoint locations. Bilinear sampling is used to obtain the correspoding scores and descriptors.
            num_patches is the number of keypoints output by this module.
        Args:
            detector_scores (torch.tensor): (b*w,1,H,W)
            weight_scores (torch.tensor): (b*w,S,H,W) Note that S=1 for scalar weights, S=3 for 2x2 weight matrices
            descriptors (torch.tensor): (b*w,C,H,W) C = descriptor dim
            keypoint_masks (torch.tensor): (b*w,1,num_patches)
        Returns:
            keypoint_coords (torch.tensor): (b*w,N,2) Keypoint locations in pixel coordinates
            keypoint_scores (torch.tensor): (b*w,S,N)
            keypoint_desc (torch.tensor): (b*w,C,N)
        """
        BW, encoder_dim, _, _ = descriptors.size()
        score_dim = weight_scores.size(1)
        detector_patches = F.unfold(detector_scores, kernel_size=self.patch_size, stride=self.patch_size)
        patch_elems = detector_patches.shape[1]

        keypoint_coords = list()
        keypoint_scores = list()
        keypoint_desc = list()
        for i in range(BW):
            v_patches_selected = torch.masked_select(self.v_patches[0], keypoint_masks[i]).view(patch_elems, -1)
            u_patches_selected = torch.masked_select(self.u_patches[0], keypoint_masks[i]).view(patch_elems, -1)
            detector_patches_selected = torch.masked_select(detector_patches[i], keypoint_masks[i]).view(patch_elems, -1)

            softmax_attention = F.softmax(detector_patches_selected, dim=0)  # (patch_elems,N)
            expected_v = torch.sum(v_patches_selected * softmax_attention, dim=0)
            expected_u = torch.sum(u_patches_selected * softmax_attention, dim=0)
            keypoint_coords1 = torch.stack([expected_u, expected_v], dim=1)  # (N,2)
            num_patches = keypoint_coords1.size(0)

            norm_keypoints2D = normalize_coords(keypoint_coords1, self.width, self.width).view(1, 1, -1, 2)

            keypoint_desc1 = F.grid_sample(descriptors[i:i+1], norm_keypoints2D, mode='bilinear', align_corners=True)
            keypoint_desc1 = keypoint_desc1.view(encoder_dim, num_patches)  # (C,N)

            keypoint_scores1 = F.grid_sample(weight_scores[i:i+1], norm_keypoints2D, mode='bilinear', align_corners=True)
            keypoint_scores1 = keypoint_scores1.view(score_dim, num_patches)  # (S,N)

            keypoint_coords.append(keypoint_coords1)
            keypoint_scores.append(keypoint_scores1)
            keypoint_desc.append(keypoint_desc1)

        return keypoint_coords, keypoint_scores, keypoint_desc
