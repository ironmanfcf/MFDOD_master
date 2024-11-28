from mfod.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .multimodality_hbb_dataset import MultiModalityHBBDataset

@DATASETS.register_module()
class MMLLVIPDataset(MultiModalityHBBDataset):


    METAINFO = {
        'classes': ('person'),
        'palette': [
             (0, 0, 142)
        ]
    }


    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)
