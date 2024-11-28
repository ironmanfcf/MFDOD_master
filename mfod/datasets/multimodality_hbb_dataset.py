import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mfod.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset

import copy
import functools
import gc
import logging
import pickle
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from mmengine.config import Config

@DATASETS.register_module()
class MultiModalityHBBDataset(BaseDetDataset):

    METAINFO = {'classes': ('car', 'person', 'bicycle'),  # FLIR
                'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]}

    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path='', img_ir_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg,
                         indices, serialize_data, pipeline, test_mode, lazy_init,
                         max_refetch)
        """    
        code-block:: none

                {
                    "metainfo":
                    {
                    "dataset_type": "test_dataset",
                    "task_name": "test_task"
                    },
                    "data_list":
                    [
                    {
                        "img_path": "test_img.jpg",
                        "img_path": "test_img_ir.jpg",
                        "height": 604,
                        "width": 640,
                        "ir_height": 604,
                        "ir_width": 640,
                        "instances":
                        [
                        {
                            "bbox": [0, 0, 10, 20],
                            "bbox_label": 1,
                            "mask": [[0,0],[0,10],[10,20],[20,0]],
                            "extra_anns": [1,2,3]
                        },
                        {
                            "bbox": [10, 10, 110, 120],
                            "bbox_label": 2,
                            "mask": [[10,10],[10,110],[110,120],[120,10]],
                            "extra_anns": [4,5,6]
                        }
                        ]
                    },
                    ]
                }
        """
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:

            self.coco = self.COCOAPI(local_path)
            # The order of returned `cat_ids` will not
            # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:

            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                    raw_ann_info,
                'raw_img_info':
                    raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``
            
        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img_path'], img_info['file_name'])
        img_ir_path = osp.join(self.data_prefix['img_ir_path'], img_info['file_name_ir'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_ir_path'] = img_ir_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        data_info['ir_height'] = img_info['ir_height']
        data_info['ir_width'] = img_info['ir_width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos