# Copyright (c) OpenMMLab. All rights reserved.
from mfod.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .multimodality_hbb_dataset import MultiModalityHBBDataset

@DATASETS.register_module()
class MMDVTODDataset(MultiModalityHBBDataset):


    METAINFO = {
        'classes':
        ('person','car','bicycle' ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142)

         ]
    }



    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(img_suffix=img_suffix,**kwargs)

