# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import logging
import collections.abc as cabc
from functools import partial as p

import torch.nn as nn

import src.models.spos as spos
from src.models.cnn_ops import GenericConv, StdConv, MBConv, DSConv


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.op = spos.SelectOp(
            p(StdConv, kernel=1, channels=channels),
            p(DSConv, kernel=3, expansion=0.5, channels=channels),
            p(DSConv, kernel=3, expansion=1, channels=channels),
            p(DSConv, kernel=3, expansion=2, channels=channels),
            p(MBConv, kernel=1, expansion=2, channels=channels),
            p(MBConv, kernel=3, expansion=0.5, channels=channels),
            p(MBConv, kernel=3, expansion=1, channels=channels),
            p(MBConv, kernel=3, expansion=2, channels=channels),

            nn.Identity
        )

    def forward(self, x):
        return self.op(x)


class Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, proj_3x3: bool = False, reduce: bool = True, dw_proj: bool=False):
        super().__init__()

        self.conv1 = GenericConv(in_channels, in_channels, 3, 2 if reduce else 1, padding=1, groups=in_channels)
        self.conv2 = GenericConv(in_channels, out_channels, 1, 1)
        if proj_3x3:
            self.proj = GenericConv(in_channels, out_channels, 3, 2, padding=1)
        else:
            if dw_proj:
                self.proj = nn.Sequential(
                    GenericConv(in_channels, in_channels, 3, 2 if reduce else 1, padding=1, groups=in_channels),
                    GenericConv(in_channels, out_channels, 1, 1)
                )            
            else:
                self.proj = GenericConv(in_channels, out_channels, 2 if reduce else 3, 2 if reduce else 1, padding=0 if reduce else 1)


    def forward(self, x):
        p = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + p


class ResNetModel(nn.Module):
    def __init__(self, stem=64, num_classes=10, blocks=[4,4,4,4], channels_scale=1.5, proj3x3:bool=False, reduce=[True, True, True, True], **kwargs):
        super().__init__()

        in_ch = kwargs['input_size'][1]
        dw_proj = kwargs.get('dw_proj')

        assert len(blocks) == len(reduce)

        self.stem = GenericConv(in_ch, stem, 3, padding=1)

        channels = stem
        layers = []
        for blk_idx, num_blocks in enumerate(blocks):
            for i in range(num_blocks):
                layers.append(Block(channels))

            new_channels = int(channels * channels_scale)
            layers.append(Reduction(channels, new_channels, proj3x3, reduce[blk_idx], dw_proj is not None))
            channels = new_channels

        self.layers = nn.ModuleList(layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        if isinstance(num_classes, cabc.Sequence):
            for idx, nc in enumerate(num_classes):
                setattr(self, f'classifier_{idx}', nn.Linear(channels, nc))

            self.multi_task = True
            self._task_id = None
            self._num_tasks = len(num_classes)
        else:
            self.classifier = nn.Linear(channels, num_classes)
            self.multi_task = False

    def set_task_id(self, task_id, strip=True):
        if not self.multi_task:
            raise ValueError('Trying to set a task ID on a model that does not support multiple tasks')
        if task_id < 0 or task_id >= self._num_tasks:
            raise ValueError(f'Invalid task ID: {task_id}')

        self._task_id = task_id
        num_classes = getattr(self, f'classifier_{self._task_id}').out_features
        logging.info(f"Task id set: {self._task_id} --> {num_classes} classes")
        if strip:
            for idx in range(self._num_tasks):
                if idx != self._task_id:
                    delattr(self, f'classifier_{idx}')

    def get_task_id(self):
        if not self.multi_task:
            raise ValueError('Trying to get a task ID from a model that does not support multiple tasks')
        return self._task_id

    def forward(self, x):
        y = self.stem(x)
        for l in self.layers:
            y = l(y)

        y = self.pool(y)
        y = self.flatten(y)
        if self.multi_task:
            y = getattr(self, f'classifier_{self._task_id}')(y)
        else:
            y = self.classifier(y)
        return y
