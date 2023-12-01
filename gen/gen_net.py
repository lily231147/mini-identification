from typing import List, Optional, Union

import torch
from torch import nn, Tensor
import utils
from gen.transformer import MultiheadAttention, attention
from utils import BaseConfig
from torch.nn import functional as F


class GenConfig(BaseConfig):
    r"""
    This is the configuration class to store specific configuration of YoloNet
    
    Args:
        in_channels (`int`, *optional*, defaults to 1):
            the channel of input feed to the network
        out_channels (`int`, *optional*, defaults to 400):
            the channel of feature map
        length (`int`, *optional*, defaults to 1024):
            sliding window length
        backbone (`str`, *optional*, defaults to 1024):
            specific the temporal feature extraction module, choosing from `attention`, `lstm`, and `empty`
    """
    def __init__(
        self, 
        epochs=1200, 
        lr=0.0001, 
        lr_drop=800, 
        gama=0.1, 
        batch_size=4, 
        load_his=False,
        in_channels=1,  
        out_channels=400, 
        length=1024, 
        backbone="attention") -> None:
        super().__init__(epochs, lr, lr_drop, gama, batch_size, load_his)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.backbone = backbone

class CNN_Module(nn.Module):
    """CNN提取局部特征"""

    def __init__(self, in_channels, out_channels):
        super(CNN_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.out_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.out_channels // 8, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.out_channels // 4, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x):
        return self.cnn(x)


class PositionEncoding(nn.Module):
    def __init__(self, in_channels, dropout=0.1, sine=True):
        super(PositionEncoding, self).__init__()
        if sine:
            self.position_embedding = self.PositionEmbeddingSine(in_channels, dropout)
        else:
            self.position_embedding = self.PositionEmbeddingLearned(in_channels)

    def forward(self, x):
        # x: shape (bs, out_channels, L); self.position_embedding(x):shape (out_channels, L)
        return self.position_embedding(x)

    # 基于sin函数的位置编码
    class PositionEmbeddingSine(nn.Module):
        def __init__(self, in_channels, dropout=0.1):
            # in_channels是（特征图）通道数，length是（特征图）长度，用于确定位置编码的大小
            super(PositionEncoding.PositionEmbeddingSine, self).__init__()
            self.in_channels = in_channels
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            assert self.in_channels % 2 == 0, "位置编码要求通道为复数"
            length = x.shape[-2]
            pe = torch.zeros(length, self.in_channels)  # 存储位置编码
            position = torch.arange(0, length).unsqueeze(1).float()  # 存储词与词的绝对位置
            # 计算每个通道的位置编码的分母部分
            # n^{d/(2i)}  i<self.in_channels // 2
            div_term = torch.full([1, self.in_channels // 2], 10000) \
                .pow((torch.arange(0, self.in_channels, 2) / self.in_channels).float()).float()
            # 偶数通道使用sin, 奇数通道使用cos
            pe[:, 0::2] = torch.sin(position / div_term)
            pe[:, 1::2] = torch.cos(position / div_term)
            return self.dropout(pe.to(x.device))

    # 基于学习的位置编码
    class PositionEmbeddingLearned(nn.Module):
        def __init__(self, n_dim, length=1024, dropout=0.1):
            super(PositionEncoding.PositionEmbeddingLearned, self).__init__()
            self.n_dim = n_dim
            self.length = length
            self.dropout = nn.Dropout(p=dropout)
            self.embed = nn.Embedding(length, n_dim)
            nn.init.uniform_(self.embed.weight)

        def forward(self, x):
            i = torch.arange(self.length).to(x.device)
            pe = self.embed(i)
            return self.dropout(pe)
        
class EventGenerator:
    """
    output events based on generate method
    """

    def __init__(self):
        super(EventGenerator, self).__init__()

    def prepare(self, input_cls_sequence, input_pos_sequence):
        return torch.cat([input_cls_sequence[:, :, None], input_pos_sequence], dim=-1)

    def next(self, inputs, hidden_feature, mask=None, pad_mask=None):
        raise NotImplementedError(
            "A specific model must to implement method `next()` to use `generate()`"
        )

    def greedy_search(
            self,
            input_cls_sequence: torch.LongTensor = None,
            input_pos_sequence: torch.FloatTensor = None,
            end_cls: Union[int, List[int]] = None,
            pad_cls: int = None,
            hidden_feature: torch.Tensor = None,
            max_length: int = 6):
        r"""
        use greedy search method generate the whole output from given input

        Parameters:
            input_cls_sequence (`torch.LongTensor` of shape `(bs, input_length)`):
                the sequence of cls used to build prompt for the generation
            input_pos_sequence (`torch.FloatTensor` of shape `(bs,input_length, 2)`):
                the sequence of pos used to build prompt for the generation
            end_cls (`Union[int, List[int]]`):
                the class id of <eos>, or class ids if more than 1 possible end tag
            pad_cls (`int`):
                the class id of <pad>
            hidden_feature (`torch.Tensor` of shape `(bs,length,d_model)`):
                the feature extracted by encoder from power sequence
            max_length (`int`):
                limit the length of outputs
        """
        device = input_cls_sequence.device
        end_cls = torch.tensor([end_cls]).to(device) if isinstance(end_cls, int) else torch.tensor(end_cls).to(device)
        unfinished_samples = torch.ones(input_cls_sequence.shape[0], dtype=torch.long, device=device)
        last_center = torch.zeros(input_cls_sequence.shape[0], dtype=torch.long, device=device)
        while True:
            # 1. combine input_cls_sequence and input_pos_sequence to prepare model inputs
            inputs = self.prepare(input_cls_sequence, input_pos_sequence)

            T = inputs.shape[1]

            # 3. config tgt_mask and pad_mask
            mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            pad_mask = (inputs[:, :, 0] == 0)

            # 2. get next_cls of shape `(bs,)` and next_pos of shape `(bs, 2)`
            cls, center, width = self.next(inputs, hidden_feature, mask=mask, pad_mask=pad_mask)
            next_cls_logits, next_center, next_width = cls[:, -1, :], center[:, -1, :], width[:, -1, 0]
            next_cls = torch.argmax(next_cls_logits, dim=-1)
            next_center = torch.argmax(next_center, dim=-1)
            exceed_mask = next_center <= last_center
            last_center = next_center
            next_cls[exceed_mask] = 3
            next_cls = next_cls * unfinished_samples
            next_center = next_center * unfinished_samples
            next_width = next_width * unfinished_samples

            # 3. update inputs
            input_cls_sequence = torch.cat([input_cls_sequence, next_cls[:, None]], dim=-1)
            next_pos = torch.cat([next_center[:, None] / 1023, next_width[:, None]], dim=-1).unsqueeze(1)
            input_pos_sequence = torch.cat([input_pos_sequence, next_pos], dim=-2)

            # 4. whether finish
            unfinished_samples = unfinished_samples.mul(
                next_cls.tile(end_cls.shape[0], 1).ne(end_cls.unsqueeze(1)).prod(dim=0)
            )
            if unfinished_samples.max() == 0 or input_cls_sequence.shape[-1] > max_length:
                return input_cls_sequence[:, 1:], input_pos_sequence[:, 1:, :]


class GenNet(nn.Module, EventGenerator):
    def __init__(self, d_model, n_class):
        super(GenNet, self).__init__()
        self.d_model = d_model
        self.n_class = n_class
        self.cnn_x = CNN_Module(1, self.d_model)
        self.tokenizer = nn.Linear(3, self.d_model)
        self.positionEncoding = PositionEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)
        self.classify = nn.Linear(d_model, n_class + 2)
        self.localize = MultiheadAttention(d_model, 8)
        self.width = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.detector = nn.Sequential(nn.Linear(d_model, 1), nn.Flatten(), nn.Linear(1024, 1), nn.Sigmoid())
        # loss
        self.bce = nn.BCELoss()
        self.cce = nn.CrossEntropyLoss()
        self.center_loss = nn.CrossEntropyLoss()
        self.width_loss = nn.SmoothL1Loss()

        self.l1 = nn.L1Loss()

    def next(self, inputs, hidden_feature, mask=None, pad_mask=None):
        # long long ago, a sb use conv1d here to implement the tokenizer function
        y = self.tokenizer(inputs.float())
        y = y + self.positionEncoding(y)
        y = self.transformer.decoder(y, hidden_feature, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
        y1, y2 = self.linear1(y), self.linear2(y)
        cls, center, width = self.classify(y), attention(y1, hidden_feature, hidden_feature)[1], self.width(y)
        return cls, center, width

    def _shift(self, tgt_cls: torch.Tensor, tgt_pos):
        shifted_cls = tgt_cls.new_zeros(tgt_cls.shape)
        shifted_cls[..., 1:] = tgt_cls[..., :-1].clone()
        shifted_cls[..., 0] = -1

        shifted_pos = tgt_pos.new_zeros(tgt_pos.shape)
        shifted_pos[:, 1:, :] = tgt_pos[:, :-1, :]

        return shifted_cls, shifted_pos

    def forward(self, x, targets, event_detect=False):
        """
        class id:
            -1 represents <bos>
            0 represents padding
            1~n_class represents event classes
            n_class+1 represents <eos>
        """
        device, bs, length = x.device, x.shape[0], x.shape[-1]

        # 1. get hidden_feature through encoder and then get the event exist
        x = x[:, 1:2, :]
        x = self.cnn_x(x).permute(0, 2, 1)
        x = x + self.positionEncoding(x)
        hidden_feature = self.transformer.encoder(x)
        exist_event = self.detector(hidden_feature).squeeze()
        hidden_feature = hidden_feature
        max_width = 6

        if self.training:
            max_length = max(len(target["labels"]) for target in targets)
            # 2. prepare the input and output of decoder
            tgt_exist_event = torch.zeros((bs,), dtype=torch.long, device=device)
            tgt_cls = torch.zeros((bs, max_length + 1), dtype=torch.long).to(device)
            tgt_pos = torch.zeros((bs, max_length + 1, 2), dtype=torch.float32).to(device)
            for id, target in enumerate(targets):
                boxes, labels = target["boxes"], target["labels"]
                if len(labels) == 0:
                    tgt_cls[id, 0] = self.n_class + 1
                    continue
                tgt_exist_event[id] = 1
                tgt_cls[id, 0:len(labels)] = torch.tensor([label for label in labels])
                tgt_cls[id, len(labels)] = self.n_class + 1
                tgt_pos[id, 0:len(labels), :] = torch.tensor(
                    [[(box[0] + box[1]) / 2/1023, (box[1] - box[0] + 1) / max_width] for box in boxes])
            shifted_cls, shifted_pos = self._shift(tgt_cls, tgt_pos)
            inputs = self.prepare(shifted_cls, shifted_pos)

            # 3. config tgt_mask and pad_mask
            mask = (torch.triu(torch.ones(max_length + 1, max_length + 1, device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            pad_mask = (shifted_cls == 0)

            # 4. get pred events and then calculate the loss
            cls, center, width = self.next(inputs, hidden_feature, mask=mask, pad_mask=pad_mask)
            if not event_detect:
                return self.cce(cls.reshape(-1, cls.shape[-1]), tgt_cls.reshape(-1)) + \
                       self.center_loss(center.reshape(-1, center.shape[-1]), (tgt_pos[..., 0]*1023).long().reshape(-1)) + \
                       self.width_loss(width.reshape(-1), tgt_pos[..., 1].reshape(-1)), None
            if max_length == 0:  # no event in the sample, so just calc the detection loss
                return self.bce(exist_event, tgt_exist_event.float()), None
            # only calc the identification loss of the sample with event, except the detection loss
            only = tgt_exist_event == 1
            return self.bce(exist_event, tgt_exist_event.float()) + \
                   self.cce(cls[only].reshape(-1, cls.shape[-1]), tgt_cls[only].reshape(-1)) + \
                   self.center_loss(center[only].reshape(-1, center.shape[-1]),
                                    tgt_pos[only][:, :, 0].long().reshape(-1)) + \
                   self.width_loss(width[only].reshape(-1), tgt_pos[only][:, :, 1].reshape(-1)), None
        else:
            results = [{"boxes": [], "labels": []} for _ in range(bs)]
            result_ids = range(bs)
            input_cls_sequence = torch.tensor([-1 for _ in range(bs)])[:, None].to(device)
            input_pos_sequence = torch.tensor([(0.0, 0.0) for _ in range(bs)])[:, None, :].to(device)
            if event_detect:
                exist_event = exist_event > 0.5
                if sum(exist_event) == 0:  # return empty result if no event detected in the batch
                    return None, results
                else:  # only identify sample with detected event
                    result_ids = torch.where(exist_event)[0]
                    input_cls_sequence = input_cls_sequence[exist_event]
                    input_pos_sequence = input_pos_sequence[exist_event]
                    hidden_feature = hidden_feature[exist_event]
            # 5. generate events by greedy search method
            cls, pos = self.greedy_search(input_cls_sequence, input_pos_sequence, self.n_class + 1, 0, hidden_feature)
            # 6. form the event
            for id, result_id in enumerate(result_ids):
                c, p = cls[id], pos[id]
                mask = (c > 0) & (c <= self.n_class)
                results[result_id]["labels"] = c[mask].cpu().detach().numpy()
                p0 = torch.clip(p[:, 0]*1023 - p[:, 1] * max_width / 2, 0, 1023)
                p1 = torch.clip(p[:, 0]*1023 + p[:, 1] * max_width / 2, 0, 1023)
                results[result_id]["boxes"] = torch.round(
                    torch.cat([p0[mask][:, None], p1[mask][:, None]], dim=-1)).long().cpu().detach().numpy()
            return None, results

