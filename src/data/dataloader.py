import torch
import numpy as np

class DataLoader:
    def __init__(self, train_dataset = [], test_dataset = [], batch_size = 1, val_split = 0.2, random_seed = 42, shuffle = True):
        # Creating data indices for training and validation splits:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,sampler=validation_sampler)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
