from torch.utils.data import Sampler
import torch


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        # dataset is sorted by length
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group
        self.group_size = self.batch_size * self.batches_per_group
        self.n_batches = len(self.data_source) // self.batch_size + (len(self.data_source) % self.batch_size != 0)
        self.n_groups = len(self.data_source) // self.group_size + (len(self.data_source) % self.group_size != 0)
        self.first = True
        self.flag = False

    def __iter__(self):
        if self.flag:
            self.first = False
        self.flag = True
        inds = torch.arange(len(self.data_source))
        groups = torch.arange(self.n_groups)
        if not self.first:
            groups = torch.randperm(groups.shape[0])
        for group_ind in groups:
            cur_inds = inds[group_ind * self.group_size : (group_ind + 1) * self.group_size]
            if not self.first:
                cur_inds = cur_inds[torch.randperm(cur_inds.shape[0])]
            for j in range(0, cur_inds.shape[0], self.batch_size):
                yield cur_inds[j : j + self.batch_size].tolist()

    def __len__(self):
        return self.n_batches
