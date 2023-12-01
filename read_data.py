import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class NILMDataset(Dataset):
    def __init__(self, powers_list, tags_list, boxes_list, stamps_list):
        super(NILMDataset, self).__init__()
        self.powers_list = powers_list
        self.tags_list = tags_list
        self.boxes_list = boxes_list
        self.stamps_list = stamps_list

    def __getitem__(self, index):
        powers = torch.as_tensor(self.powers_list[index], dtype=torch.float32)
        tags= torch.as_tensor(self.tags_list[index], dtype=torch.int64)
        boxes = torch.as_tensor(self.boxes_list[index], dtype=torch.float32)
        stamps = torch.as_tensor(self.stamps_list[index], dtype=torch.float64)
        return powers, {"labels": tags, "boxes": boxes}, stamps
    
    def __len__(self):
        return len(self.tags_list)

    def skip_no_events(self):
        mask = [len(tags)>0 for tags in self.tags_list]
        self.powers_list = [powers for powers, flag in zip(self.powers_list, mask) if flag]
        self.tags_list = [tags for tags, flag in zip(self.tags_list, mask) if flag]
        self.boxes_list = [boxes for boxes, flag in zip(self.boxes_list, mask) if flag]
        self.stamps_list = [stamps for stamps, flag in zip(self.stamps_list, mask) if flag]

    @staticmethod
    def collate_fn(batch):
        powers, targets, stamps = tuple(zip(*batch))
        return torch.stack(powers)[:, None, :], targets, torch.stack(stamps)

def get_datasets(set_dir, set_name, app_idx, length=1024, ratio=(2, 1, 1), load_difficult=False):
    powers_list, tags_list, boxes_list, stamps_list = load_data(set_dir, set_name, app_idx, length, load_difficult)
    positive_events_ids = [idx for idx, tags in enumerate(tags_list) if len(tags) > 0]
    val_start = positive_events_ids[len(positive_events_ids) * ratio[0] // sum(ratio)]
    test_start = positive_events_ids[len(positive_events_ids) * (ratio[0]+ratio[1]) // sum(ratio)]
    train_set = NILMDataset(powers_list[:val_start], tags_list[:val_start], boxes_list[:val_start], stamps_list[:val_start])
    val_set = NILMDataset(powers_list[val_start:test_start], tags_list[val_start:test_start], boxes_list[val_start:test_start], stamps_list[val_start:test_start])
    test_set = NILMDataset(powers_list[test_start:], tags_list[test_start:], boxes_list[test_start:], stamps_list[test_start:])
    train_set.skip_no_events()
    return train_set, val_set, test_set

def load_data(set_dir, set_name, app_idx, length=1024, load_difficult=False):
    r"""
    Load data from disk. The original is slided into sliding windows based on the `length` and `stride`, 
    where we skip the windows with intervals or difficult events
    
    Returns:
        powers_list (List[NDArray[float32]]): 
            a list of powers, where the powers of shape `(length)` is known as example 
        tags_list (List[NDArray[int]]):
            a list of tags, where the tags of shape `(n_events_in_example)` is the event labels within an example
        boxes_list (List(NDArray[int])):
            a list of boxes, where the boxes of shape `(n_events_in_example, 2)` is the position corresponding to tags
        stamps_list (List[NDArray[float64]]):
            a list of stamps where the stamps of shape `(length)` is the timestamps corresponding to powers
    """
    if set_name == 'redd':
        # for redd, only channel 1 is used as mains since it performs poorly when use both channel 1 and channel 2
        mains = np.loadtxt(set_dir / "channel_1.dat", usecols=(0, 1))
    elif set_name == 'ukdale':
        mains = np.loadtxt(set_dir / "mains.dat", usecols=(0, 1))
    else:
        raise ValueError("set name must in [`ukdale`, `redd`]")
    all_labels = np.loadtxt(set_dir / f"channel_{str(app_idx)}" / "labels.txt")
    intervals = np.loadtxt(set_dir / f"channel_{str(app_idx)}" / "interval.txt")
    start = 0
    powers_list = []
    tags_list = []
    boxes_list = []
    stamps_list = []
    while start + length <= len(mains):
        stamps, powers = (mains_in_window:=mains[start: start + length])[:, 0], mains_in_window[:, 1]
        # skip the interval
        in_interval, interval_end = skip_interval(stamps, intervals)
        if in_interval:
            start = np.argwhere(mains[:, 0] > interval_end)[0][0]
            continue
        # get the labels in the sliding window, win_labels: (None, 5)
        labels_in_window = all_labels[(stamps[0] <= all_labels[:, 0]) & (all_labels[:, 1] <= stamps[-1])]
        # skip if you don't consider but there exist the difficult event
        if not load_difficult and np.any(labels_in_window[:, 4]):
            start = np.argwhere(mains[:, 0] > labels_in_window[np.argwhere(labels_in_window[:, 4] == 1)[-1][0], 1])[0][0]
            continue
        # generate powers and targets
        powers_list.append(powers)
        tags_list.append(labels_in_window[:, 3])
        boxes_list.append(np.array([[np.where(stamps == time)[0][0] for time in record] for record in labels_in_window[:, :2]]))
        stamps_list.append(stamps)
        start += length
    return powers_list, tags_list, boxes_list, stamps_list

def skip_interval(stamps, intervals):
    for interval in intervals:
        # there is a cross between power and interval
        if max(stamps[0], interval[0]) < min(stamps[-1], interval[1]):
            return True, interval[1]
    return False, None