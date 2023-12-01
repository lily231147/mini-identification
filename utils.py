import json
import os
from typing import Union, List
import sys
sys.path.append("/home/aistudio/external-libraries")

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class BaseConfig:
    r"""
    This is the configuration class to store some basic configuration

    Args:
        epochs (`int`, *optional*, defaults to 1000):
            total number of training rounds.
        lr (`float`, *optional*, defaults to 1e-4):
            init learning rate
        lr_drop (`int`, *optional*, defaults to 400):
            interval for updating learning rate
        gama (`float`, *optional*, defaults to 1e-1):
            factor for updating learning rate
        batch_size (`int`, *optional*, defaults to 4):
            the size of each batch
        load_his (`bool`, *optional*, defaults to False):
            whether load checkpoint or train from scratch
    """
    def __init__(
        self,
        epochs=1000,
        lr = 1e-4,
        lr_drop=400,  
        gama=0.1, 
        batch_size=4,
        load_his=False,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.lr_drop = lr_drop
        self.gama = gama
        self.batch_size = batch_size
        self.load_his = load_his
    
    def save(self, path: Path):
        with open(path, "w") as file:
            json.dump(self.__dict__, file, indent=4)
    
    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as file:
            return cls(**json.load(file)) 


"""
由targets（包含样本中事件的起止位置和类别）生成按序列标注：
输出(L,N)
"""


def to_seq(targets, length, device, label_method="BIO"):
    ground = torch.zeros((len(targets), length), dtype=torch.long).to(device)
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        for box_idx, box in enumerate(boxes):
            if label_method == "BO":
                ground[batch_idx, int(box[0]): int(box[1]) + 1] = labels[box_idx]
            elif label_method == "BIO":
                ground[batch_idx, int(box[0])] = 2 * labels[box_idx] - 1
                ground[batch_idx, int(box[0]) + 1: int(box[1]) + 1] = 2 * labels[box_idx]
            elif label_method == "BILO":
                ground[batch_idx, int(box[0])] = 3 * labels[box_idx] - 2
                ground[batch_idx, int(box[0]) + 1: int(box[1])] = 3 * labels[box_idx] - 1
                ground[batch_idx, int(box[1])] = 3 * labels[box_idx]
    return ground


"""
输入2个锚框序列，boxes1有N个框，boxes2有M个框
输出(N,M),iou[i,j]是boxes1中i框和boxes2中j框的iou
"""


def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # right-bottom [N,M,2]
    inter = (rb - lt).clamp(min=0)  # [N,M]，inter[i,j]是boxes1中i框和boxes2中j框的相交区域，不相交设为0
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def cal_metrics(preds, targets, powers, case_dir, enable_record=False):
    tp = 0
    fp = 0
    fn = 0
    for pred, target, power in zip(preds, targets, powers):
        pred_boxes = torch.tensor(pred["boxes"]).to(torch.device("cuda"))
        target_boxes = target["boxes"]
        pred_labels = torch.tensor(pred["labels"]).to(torch.device("cuda"))
        target_labels = target["labels"]
        candidate_classes = torch.unique(torch.cat([pred_labels, target_labels]))
        is_record = False
        for clz in candidate_classes:
            mask_pred = pred_labels == clz
            mask_target = target_labels == clz
            pred_boxes_clz = pred_boxes[mask_pred]
            target_boxes_clz = target_boxes[mask_target]
            n_pred_boxes = len(pred_boxes_clz)
            n_gt_boxes = len(target_boxes_clz)
            selected = []
            if n_pred_boxes == 0 or n_gt_boxes == 0:
                fn += n_gt_boxes
                fp += n_pred_boxes
                is_record = is_record if n_pred_boxes == 0 and n_gt_boxes == 0 else True
                continue
            ious = box_iou(target_boxes_clz, pred_boxes_clz)
            # matches: shape(len(pred_boxes)) 表示每个预测框对应的真实框索引，没有则为-1
            matxhes_val, matches = torch.max(ious, dim=0)
            matches[matxhes_val < 0.03] = -1
            for idx, match in enumerate(matches):
                if match >= 0:
                    if match not in selected:
                        tp += 1
                        selected.append(match)
                else:
                    fp += 1
                    is_record = True
            fn += (fn_temp := len(target_boxes_clz) - len(torch.unique(matches[matches >= 0])))
            is_record = is_record if fn_temp==0 else True
        if enable_record and is_record:
            recording(power, pred, target, case_dir)
    return tp, fp, fn


class Metric:
    def __init__(self):
        super(Metric, self).__init__()
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def get_index(self):
        pre = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
        f1 = 2.0 * pre * rec / (pre + rec) if pre + rec > 0 else 0
        return pre, rec, f1

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


# 过滤不满足规则的预测事件
def filter_pred(app_idx, preds, powers, stamps, intervals, amplitude_threshold, stable_threshold):
    device_list = [9, 15]
    intervals = torch.tensor(intervals).to(powers.device)
    results = [{"boxes": [], "labels": []} for _ in range(len(preds))]
    for idx, (pred, power, stamp) in enumerate(zip(preds, powers, stamps)):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        for label, box in zip(pred_labels, pred_boxes):
            start, end = box[0], box[1]
            if end - start < 1:
                continue
            # 计算是功率是否在区间内，不在功率区间内的事件会被扔出去
            event = power[start:end + 1]
            amplitude = (torch.max(event) - torch.min(event))  # 预测区间的功率幅值
            x = amplitude_threshold['max'][label - 1]
            y = amplitude_threshold['min'][label - 1]
            if amplitude > x or amplitude < y:
                continue
            # 计算是否在断裂区间，在的事件会被扔出去
            # if torch.any(torch.max(intervals[:, 0], stamp[start]) < torch.min(intervals[:, 1], stamp[end])):
            #     continue
            # redd16号用电器的四事件是一个凸起，始末功率接近
            if app_idx == 16 and label == 4 and torch.abs(power[start] - power[end]) >= 30:
                continue
            # 判断趋势是否稳定
            if app_idx in device_list:
                if box[1] - box[0] < 1 \
                        or torch.abs(power[start] - power[start + 1]) > stable_threshold['front'][label - 1] \
                        or torch.abs(power[end] - power[end - 1]) > stable_threshold['end'][label - 1]:
                    continue
            results[idx]["labels"].append(label)
            results[idx]["boxes"].append([start, end])
    return results


def recording(power, pred, target, case_dir):
    power = power.cpu().detach().numpy()
    target = {k: v.cpu().detach().numpy().tolist() for k, v in target.items()}
    # pred = {k: v.cpu().detach().numpy().tolist() for k, v in pred.items()}
    plt.figure(figsize=(16, 10), dpi=300)
    plt.plot(power)
    max_power = max(power)
    for label, boxes in zip(pred['labels'], pred['boxes']):
        plt.axvline(boxes[0].item(), ymin=max_power/2, color='lightblue')
        plt.axvline(boxes[1].item(), ymin=max_power/2, color='blue')
        plt.text(boxes[1].item(), max_power, str(label.item()), color='blue')
    for label, boxes in zip(target['labels'], target['boxes']):
        plt.axvline(boxes[0], ymax=max_power/2, color='lightcoral')
        plt.axvline(boxes[1], ymax=max_power/2,color='red')
        plt.text(boxes[1], max_power/2, str(label), color='red')
    plt.savefig(f"{case_dir}/{target['idx']}.png", bbox_inches='tight')
    plt.cla()
    
# def convert_label():
#     set_dir = Path(f"data/source/ukdale")
#     all_labels = np.loadtxt(set_dir / f"channel_15" / "labels.txt", dtype='str')
#     all_labels[:, 3][all_labels[:, 3] == '1'] = '2'
#     all_labels[:, 3][all_labels[:, 3] == '0'] = '1'
#     np.savetxt('back.txt', all_labels, fmt='%s')
# convert_label()