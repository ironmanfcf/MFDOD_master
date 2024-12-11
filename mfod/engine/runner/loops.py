# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop
from mmengine.runner import EpochBasedTrainLoop
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader

from mfod.registry import LOOPS
from ...adm.optim import *



@LOOPS.register_module()
class GMTAEpochBasedTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
            super().__init__(runner,
                             dataloader,
                             max_epochs,
                             val_begin,
                             val_interval,
                             dynamic_intervals)
            self.balancer = get_method("amtl", "min") #amtl的定义

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        
        
        
        shared_parameter = []
        for name, p in self.runner.model.backbone.named_parameters():
            if p.requires_grad:
                shared_parameter.append(p)
        for name, p in self.runner.model.neck.named_parameters():
            if p.requires_grad:
                shared_parameter.append(p)
        fusion_parameter = self.runner.model.fusion #各自的参数
        detection_parameter_1 = self.runner.model.roi_head  #检测网络的参数 
        detection_parameter_2 = self.runner.model.rpn_head  #检测网络的参数 
        detection_parameter_1 = list(detection_parameter_1.parameters())
        detection_parameter_2 = list(detection_parameter_2.parameters())
        combined_parameter = detection_parameter_1 + detection_parameter_2
        task_specific_params={'0' : list(fusion_parameter.parameters()), '1' : combined_parameter}
        
        outputs = self.runner.model.train_step(
            data_batch, shared_parameter, task_specific_params,iter=self._iter,optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1