import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRefMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.sparse = config['networks']['matcher_block']['sparse']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.width**2).float()  # HW
        u_coord = u_coord.reshape(self.width**2).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        self.src_coords_dense = coords.unsqueeze(0).to(self.gpuid)  # 1 x HW x 2

    def forward(self, keypoint_scores, keypoint_desc, desc_dense, keypoint_coords):
        """
        Args:
            keypoint_scores: (b*w,S,N)
            keypoint_desc: (b*w,C,N)
            desc_dense: (b*w,C,H,W)
            keypoint_coords: (b*w, N, 2)
        Returns:
            pseudo_coords (torch.tensor): (b*(w-1),N,2)
            match_weights (torch.tensor): (b*(w-1),S,N)
            tgt_ids (torch.tensor): (b*(w-1),) indices along batch dimension for target data
            src_ids (torch.tensor): (b*(w-1),) indices along batch dimension for source data
        """
        BW = len(keypoint_desc)
        encoder_dim = keypoint_desc[0].shape[0]
        B = int(BW / self.window_size)
        src_desc_dense = desc_dense[::self.window_size]
        src_desc_unrolled = F.normalize(src_desc_dense.view(B, encoder_dim, -1), dim=1)  # B x C x HW
        # build pseudo_coords
        pseudo_coords = list()
        tgt_ids = torch.zeros(B * (self.window_size - 1), dtype=torch.int64, device=self.gpuid)    # B * (window - 1)
        src_ids = torch.zeros(B * (self.window_size - 1), dtype=torch.int64, device=self.gpuid)    # B * (window - 1)
        # loop for each batch
        if not self.sparse:
            for i in range(B):
                win_ids = torch.arange(i * self.window_size + 1, i * self.window_size + self.window_size).to(self.gpuid)
                for win_id in win_ids:
                    tgt_desc = keypoint_desc[win_id]
                    tgt_desc = F.normalize(tgt_desc, dim=0)
                    match_vals = torch.matmul(tgt_desc.transpose(1, 0), src_desc_unrolled[i])  # N x HW
                    soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=1)  # N x HW
                    pseudo_id = win_id - (win_id // self.window_size + 1)
                    pseudo_coords.append(torch.matmul(self.src_coords_dense[0].transpose(1, 0),
                                                            soft_match_vals.transpose(1, 0)).transpose(1, 0))  # N x 2
                    tgt_ids[pseudo_id] = win_id
                    src_ids[pseudo_id] = i * self.window_size
        else:
            for i in range(B):
                win_ids = torch.arange(i * self.window_size + 1, i * self.window_size + self.window_size).to(self.gpuid)
                for win_id in win_ids:
                    tgt_desc = keypoint_desc[win_id]
                    src_desc = keypoint_desc[i*self.window_size]
                    tgt_desc = F.normalize(tgt_desc, dim=0)
                    src_desc = F.normalize(src_desc, dim=0)
                    match_vals = torch.matmul(tgt_desc.transpose(1, 0), src_desc)
                    soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=1)
                    src_coords = keypoint_coords[i*self.window_size]
                    pseudo_id = win_id - (win_id // self.window_size + 1)
                    pseudo_coords.append(torch.matmul(src_coords.transpose(1, 0), soft_match_vals.transpose(1, 0)).transpose(1, 0))
                    tgt_ids[pseudo_id] = win_id
                    src_ids[pseudo_id] = i * self.window_size

        return pseudo_coords, [keypoint_scores[tgt_id] for tgt_id in tgt_ids], tgt_ids, src_ids
