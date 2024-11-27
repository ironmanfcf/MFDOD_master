import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mfod.registry import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class LLVIPDataset(CocoDataset):
    """Dataset for LLVIP."""

    METAINFO = {
        'classes': ('person'),
        'palette': [
             (0, 0, 142)
        ]
    }

