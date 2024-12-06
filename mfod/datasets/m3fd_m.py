
from mfod.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .multimodality_hbb_dataset import MultiModalityHBBDataset

@DATASETS.register_module()
class MMM3FDDataset(MultiModalityHBBDataset):


    METAINFO = {
        'classes':
        ('People','Car','Truck','Bus','Motorcycle','Lamp'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(img_suffix=img_suffix,**kwargs)