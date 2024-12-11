from mmengine.optim import OptimWrapper

from typing import Optional
import torch
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from typing import Dict, List, Optional


from mfod.registry import OPTIM_WRAPPERS
from ...adm.optim import *

@OPTIM_WRAPPERS.register_module()
class GMTAOptimWrapper(OptimWrapper):
    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        super().__init__(optimizer=optimizer,
                         accumulative_counts=accumulative_counts,
                         clip_grad=clip_grad)
        self.balancer = get_method("amtl", "min")  # amtl 的定义


    def gmta_update_params(  # type: ignore
            self,
            parsed_losses,
            shared_parameter, 
            task_specific_params,
            iter,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters by amtl`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """

        # self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
       
        self.balancer.step_with_model(
                    losses = parsed_losses,
                    shared_params = shared_parameter,
                    task_specific_params = task_specific_params,
                    last_shared_layer_params = None,
                    iter=iter
                )
        if self.should_update():              
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)
            
    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.zero_grad`.
        """
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)
