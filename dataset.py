import torch
import datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

#%%
class BartDataset(Dataset):
    def __init__(self, val=False):
        self.val = val
        if self.val:
            self.dataset = datasets.load_from_disk('cnndm/full/validation').train_test_split(0.96, shuffle=False, seed=1)['train']
        else:
            self.dataset = datasets.load_from_disk('cnndm/full/train').train_test_split(0.96, shuffle=False, seed=1)['train']

    def __getitem__(self, idx):
        if self.val:
            input_ids = torch.tensor(self.dataset[idx]['input_ids'])
            attention_mask = torch.tensor(self.dataset[idx]['attention_mask'])
            decoder_input_ids = torch.tensor(self.dataset[idx]['decoder_input_ids'])
            decoder_attention_mask = torch.tensor(self.dataset[idx]['decoder_attention_mask'])
            labels = torch.tensor(self.dataset[idx]['labels'])
            
            return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels
            
        else:
            positive_masks = torch.tensor(self.dataset[idx]['positive_masks'])
            negative_masks = torch.tensor(self.dataset[idx]['negative_masks'])
            input_ids = torch.tensor(self.dataset[idx]['input_ids'])
            attention_mask = torch.tensor(self.dataset[idx]['attention_mask'])
            decoder_input_ids = torch.tensor(self.dataset[idx]['decoder_input_ids'])
            decoder_attention_mask = torch.tensor(self.dataset[idx]['decoder_attention_mask'])
            labels = torch.tensor(self.dataset[idx]['labels'])
            
            return positive_masks, negative_masks, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels
    
    def __len__(self):
        return len(self.dataset)


def get_loader(batch_size, num_workers, model_name):    
    train_loader = DataLoader(dataset=BartDataset(),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=BartDataset(val=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader


def get_dist_loader(batch_size, num_workers, model_name):
    train_dataset = BartDataset()
    val_dataset = BartDataset(val=True)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              pin_memory=True,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=val_sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=num_workers)
    
    return train_loader, val_loader, train_sampler, val_sampler