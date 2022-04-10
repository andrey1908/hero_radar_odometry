import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler
from utils.utils import get_transform

class RadiateDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.data_dir = config['data_dir']
        sequences = sorted(os.listdir(self.data_dir))
        self.sequences = self.get_sequences_split(sequences, split)
        self.seq_idx_range = {}
        self.seq_lens = []
        self.frames = []
        self.stamps = []
        for seq in self.sequences:
            seq_frames = sorted(os.listdir(os.path.join(self.data_dir, seq, 'Navtech_Cartesian')))
            self.seq_idx_range[seq] = [len(self.frames), len(self.frames) + len(seq_frames)]
            self.seq_lens.append(len(seq_frames))
            self.frames.extend(seq_frames)
            with open(os.path.join(self.data_dir, seq, 'Navtech_Cartesian.txt')) as f:
                lines = f.readlines()
            curr_frame = 1
            for line in lines:
                split_line = line[:-1].split()
                assert split_line[0] == "Frame:" and split_line[2] == "Time:" and int(split_line[1]) == curr_frame
                self.stamps.append(int(float(split_line[3]) * 10**6))
                curr_frame += 1
        assert len(self.frames) == len(self.stamps)

        self.radar_resolution = 0.175
        self.cart_resolution = self.config['cart_resolution']
        self.cart_pixel_width = self.config['cart_pixel_width']

    def get_sequences_split(self, sequences, split):
        self.split = self.config['train_split']
        if split == 'validation':
            self.split = self.config['validation_split']
        elif split == 'test':
            self.split = self.config['test_split']
        return [seq for i, seq in enumerate(sequences) if i in self.split]

    def get_seq_from_idx(self, idx):
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def mean_intensity_polar_mask(self, polar_data):
        multiplier = 2.0
        range_bins, num_azimuths = polar_data.shape
        polar_mask = np.zeros((range_bins, num_azimuths))
        for i in range(num_azimuths):
            m = np.mean(polar_data[:, i])
            polar_mask[:, i] = polar_data[:, i] > multiplier * m
        return polar_mask.astype(np.float32)

    def polar_to_cartesian(self, polar):
        width = (self.cart_pixel_width - 1) * self.cart_resolution
        coords = np.linspace(-width / 2, width / 2, self.cart_pixel_width, dtype=np.float32)
        Y, X = np.meshgrid(coords, -1 * coords)

        sample_range = np.sqrt(Y * Y + X * X)
        sample_angle = np.arctan2(Y, X)
        sample_angle += (sample_angle < 0).astype(np.float32) * 2 * np.pi

        azimuth_step = 2 * np.pi / polar.shape[1]
        sample_u = sample_angle / azimuth_step - 0.5
        sample_v = sample_range / self.radar_resolution - 0.5

        polar = np.concatenate((polar[:,-1:], polar, polar[:,:1]), axis=1)
        sample_u = sample_u + 1
        sample_v[sample_v < 0] = 0

        polar_to_cart_warp = np.stack((sample_u, sample_v), axis=-1)
        return np.expand_dims(cv2.remap(polar, polar_to_cart_warp, None, cv2.INTER_LINEAR), axis=0)

    def augment_batch(self, batch):
        rot_max = self.config['augmentation']['rot_max']
        data_batch = batch['data'].numpy()
        mask_batch = batch['mask'].numpy()
        polar_batch = batch['polar'].numpy()

        BW, _, n_azimuth = polar_batch.shape
        azimuth_resolution = 2 * np.pi / n_azimuth
        T_aug = list()
        for i in range(BW):
            if i % self.config['window_size'] == 0:
                continue
            # only for target frames
            polar_data = polar_batch[i].squeeze()
            rot = np.random.uniform(-rot_max, rot_max)
            rot_azimuths = int(np.round(rot / azimuth_resolution))
            rot = rot_azimuths * azimuth_resolution

            polar_data = np.roll(polar_data, rot_azimuths, axis=1)
            polar_mask = self.mean_intensity_polar_mask(polar_data)
            data = self.polar_to_cartesian(polar_data)
            mask = self.polar_to_cartesian(polar_mask)

            data_batch[i] = data
            mask_batch[i] = mask
            T_aug += [torch.from_numpy(get_transform(0, 0, rot))]
            polar_batch[i] = polar_data

        batch['data'] = torch.from_numpy(data_batch)
        batch['polar'] = torch.from_numpy(polar_batch)
        batch['mask'] = torch.from_numpy(mask_batch)
        batch['T_aug'] = T_aug

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = os.path.join(self.data_dir, seq, 'Navtech_Polar', self.frames[idx])

        polar_data = cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        polar_mask = self.mean_intensity_polar_mask(polar_data)
        data = self.polar_to_cartesian(polar_data)
        mask = self.polar_to_cartesian(polar_mask)

        t_ref = np.array([self.stamps[idx], -1]).reshape(1, 2)

        timestamps = np.arange(self.stamps[idx], self.stamps[idx] + 0.25 * 10**6, 0.25 * 10**6 / 400, dtype=np.int)
        timestamps = np.expand_dims(np.expand_dims(timestamps, axis=0), axis=-1)

        return {'data': data, 't_ref': t_ref, 'mask': mask,
                'timestamps': timestamps, 'polar': polar_data}

def get_dataloaders_radiate(config):
    """Returns the dataloaders for training models in pytorch.
    Args:
        config (json): parsed configuration file
    Returns:
        train_loader (DataLoader)
        valid_loader (DataLoader)
        test_loader (DataLoader)
    """
    vconfig = dict(config)
    vconfig['batch_size'] = 1
    train_dataset = RadiateDataset(config, 'train')
    valid_dataset = RadiateDataset(vconfig, 'validation')
    test_dataset = RadiateDataset(vconfig, 'test')
    train_sampler = RandomWindowBatchSampler(config['batch_size'], config['window_size'], train_dataset.seq_lens)
    valid_sampler = SequentialWindowBatchSampler(1, config['window_size'], valid_dataset.seq_lens)
    test_sampler = SequentialWindowBatchSampler(1, config['window_size'], test_dataset.seq_lens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader
