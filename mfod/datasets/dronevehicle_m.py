import glob
import os.path as osp
from typing import List, Union, Optional, Sequence, Mapping, Callable

from mmengine.fileio import get_local_path
from mfod.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .multimodality_obb_dataset import MultiModalityOBBDataset

@DATASETS.register_module()
class MMDroneVehicleDataset(MultiModalityOBBDataset):

    METAINFO = {
        'classes': ('car', 'truck', 'bus', 'van', 'feright_car'),
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0), (138, 43, 226)]
    }
    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)