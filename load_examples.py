import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

"""
读取一个样本的xml标签信息
输入： xml路径
输出：1. 样本中所有事件的类别列表，形状为：[1,2,...]
     2. 样本中所有事件的起止位置，形状为：[[1,2],[3,4],...]
     3. 样本是否存在困难事件
"""


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # class_list存放样本每个事件的类别；box_list存放样本每个事件的位置；flag标记是否为困难样本
    class_list, box_list, flag = [], [], 0
    for child in root:
        # 每一个object对应于一个事件
        if child.tag != 'object':
            continue
        for item in child:
            if item.tag == 'name':
                # 事件类型
                class_list.append(int(item.text.split('-')[-1]))
            if item.tag == 'bndbox':
                # 事件位置
                box_list.append([int(node.text) for node in item])
            if item.tag == 'difficult' and flag == 0:
                # 只要样本存在困难事件，则标记该样本为困难
                flag = int(item.text)
    return class_list, box_list, flag


"""
输入：1. app_idx，设备号
     2. 所在数据集名称（redd或ukdale）
     3. partition，数据集中训练样本截止索引、验证样本截止索引、测试样本截止索引，eg：[322, 374, 1393]
     4. flag：数据集标签（train，val或test）
输出：样本列表，列表中每个样本的信息也组织为一个列表，分别是：
    1. 样本索引：该样本在数据集中的索引，主要用于错例分析
    2. 功率序列：样本对应的总线功率序列，形状为[[1,2,...]]
    3. 所有事件的类别列表：输出形式同parse_xml()
    4. 所有事件的位置列表：输出形式同parse_xml()
"""


def read_data(app_idx, dataset, partition, flag="train", only_events=True):
    # 设备的数据主目录，eg: ./data/ukdale/data8/
    path = './data/examples/' + dataset + '/data%d/' % app_idx
    # 训练集只考虑有目标设备事件的样本，而验证集&测试集考虑所有样本
    # ps: 这样会导致无法利用训练集中无目标设备事件的样本，但是无所谓了，影响不大，如确实需要可以把这部分样本放到验证集中
    if flag == "train":
        # 读取训练集中有目标设备事件的样本的索引，eg：./data/ukdale/data8/ImageSets/train.txt
        if only_events:
            record_idxs = np.loadtxt(path + 'ImageSets/' + flag + '.txt', dtype=int)
        else:
            event_sample_idxs = np.loadtxt(path + 'ImageSets/' + flag + '.txt', dtype=int)
            n_event_sample = len(event_sample_idxs)
            record_idxs = np.arange(0, partition[0] + 1)
            record_idxs = np.union1d(
                np.random.choice(np.setdiff1d(record_idxs,event_sample_idxs),n_event_sample,replace=False),
                event_sample_idxs)
    elif flag == "val":
        record_idxs = np.arange(partition[0] + 1, partition[1] + 1)
    else:
        record_idxs = np.arange(partition[1] + 1, partition[2] + 1)

    event_powers_num = 0 # 仅用于记录事件个数，用于打印日志
    event_num = 0  # 仅用于记录事件个数，用于打印日志
    samples = []  # 存放样本信息
    for idx in record_idxs:
        # 读取标签
        labels, boxes, is_difficult = parse_xml(os.path.join(path, 'Annotations', str(idx) + '.xml'))
        if is_difficult == 1:
            # 不处理存在difficute事件的样本
            continue
        # 样本功率序列
        power = np.loadtxt(os.path.join(path, 'JPEGImages', str(idx) + '.txt'))
        # 样本信息：样本索引，功率序列，所有事件的类别列表，所有事件的位置列表
        samples.append([idx, power[:, 0], power[:, 1], labels, boxes])
        event_powers_num += 1 if len(labels) > 0 else 0
        event_num += len(labels)
    print(f"{flag}数据已加载，共{len(samples)}个样本，共{event_powers_num}个样本存在目标事件, 共{event_num}个目标事件")
    return samples


class NILMDataSet(Dataset):
    def __init__(self, app_idx, set_name, partition, type):
        # samples中每个sample包括：样本索引，功率信号，样本类别，事件位置
        samples = read_data(app_idx, set_name, partition, type)
        # 将样本索引，功率信号，样本类别，事件位置拆分出来
        self.idx_list, self.stamp_list, self.power_list, self.labels_list, self.boxes_list = zip(*samples)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        """
        返回一个样本的相关数据
        idx: 样本索引，int
        power: 功率数据，shape:(2,L), L是窗口长度,存储时间戳序列和功率序列
        labels: 样本内各事件的类别，shape: (m,), m是样本内事件数量
        boxes: 样本内各事件的起止位置，shape: (m,2)
        """
        idx = torch.as_tensor(self.idx_list[index])
        stamp = torch.as_tensor(self.stamp_list[index], dtype=torch.float64)
        power = torch.as_tensor(self.power_list[index], dtype=torch.float32)[None, None, :]
        labels = torch.as_tensor(self.labels_list[index], dtype=torch.int64)
        boxes = torch.as_tensor(self.boxes_list[index], dtype=torch.float32)
        return power, {"idx": idx, "stamp": stamp, "boxes": boxes, "labels": labels}

    @staticmethod
    def collate_fn(batch):
        """
        __getitem__返回值的target是一个字典，不能直接使用默认的方法合成batch
        没办法用torch.cat()将一个batch的字典堆叠在一起，需要手动将其组合成一个元组
        """
        images, targets = tuple(zip(*batch))
        return torch.cat(images, dim=0), targets


def get_loaders(app_idx, set_name, batch_size, partition):
    train_dataset = NILMDataSet(app_idx, set_name, partition, "train")
    val_dataset = NILMDataSet(app_idx, set_name, partition, "val")
    test_dataset = NILMDataSet(app_idx, set_name, partition, "test")
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=NILMDataSet.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=360, collate_fn=NILMDataSet.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=NILMDataSet.collate_fn)
    return train_loader, val_loader, test_loader
