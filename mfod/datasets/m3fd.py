import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset




@DATASETS.register_module()
class M3FDDataset(CocoDataset):
    """Dataset for M3FD."""

    METAINFO = {
        'classes':
        ('People','Car','Truck','Bus','Motorcycle','Lamp'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
