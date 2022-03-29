import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets.custom_sampler import RandomWindowBatchSampler, SequentialWindowBatchSampler

class RadiateDataset(Dataset):

    """Radiate Dataset"""
    
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

        self.radiate_cart_resolution = 0.175

    def get_sequences_split(self, sequences, split):
        """Retrieves a list of sequence names depending on train/validation/test split.
        Args:
            sequences (List[AnyStr]): list of all the sequences, sorted lexicographically
            split (List[int]): indices of a specific split (train or val or test) aftering sorting sequences
        Returns:
            List[AnyStr]: list of sequences that belong to the specified split
        """
        self.split = self.config['train_split']
        if split == 'validation':
            self.split = self.config['validation_split']
        elif split == 'test':
            self.split = self.config['test_split']
        return [seq for i, seq in enumerate(sequences) if i in self.split]

    def get_seq_from_idx(self, idx):
        """Returns the name of the sequence that this idx belongs to.
        Args:
            idx (int): frame index in dataset
        Returns:
            AnyStr: name of the sequence that this idx belongs to
        """
        for seq in self.sequences:
            if self.seq_idx_range[seq][0] <= idx and idx < self.seq_idx_range[seq][1]:
                return seq
        assert(0), 'sequence for idx {} not found in {}'.format(idx, self.seq_idx_range)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.get_seq_from_idx(idx)
        frame = os.path.join(self.data_dir, seq, 'Navtech_Cartesian', self.frames[idx])

        data = cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scale = self.radiate_cart_resolution / self.config['cart_resolution']
        scaled_size = (int(data.shape[1] * scale), int(data.shape[0] * scale))
        data = cv2.resize(data, scaled_size, interpolation=cv2.INTER_LINEAR)

        width = self.config['cart_pixel_width']
        shift_0 = data.shape[0] // 2 - width // 2
        shift_1 = data.shape[1] // 2 - width // 2
        if shift_0 < 0 or shift_1 < 0 or shift_0 + width > data.shape[0] or shift_1 + width > data.shape[1]:
            raise RuntimeError("Can not transform radar image")
        data = data[shift_0: shift_0 + width, shift_1: shift_1 + width]
        data = np.expand_dims(data, axis=0)

        t_ref = np.array([self.stamps[idx], -1]).reshape(1, 2)

        mask = (data > np.mean(data)).astype(np.float32)

        timestamps = np.arange(self.stamps[idx], self.stamps[idx] + 0.25 * 10**6, 0.25 * 10**6 / 400, dtype=np.int)
        timestamps = np.expand_dims(np.expand_dims(timestamps, axis=0), axis=-1)

        return {'data': data, 't_ref': t_ref, 'mask': mask,
                'timestamps': timestamps}

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
