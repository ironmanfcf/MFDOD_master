from mfod.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .multimodality_obb_dataset import MultiModalityOBBDataset


@DATASETS.register_module()
class MMVEDAIDataset(MultiModalityOBBDataset):

    METAINFO = {
        'classes':
        ( 'van','camping_car','car','pick-up', 'truck','tractor', 'boat','vehicle', 'plane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(img_suffix=img_suffix, **kwargs)