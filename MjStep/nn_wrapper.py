import torch
import torch.nn as nn
import mujoco

from typing import Tuple

from MjStep.autograd_mujoco import pyMjStep
import torchmj

class MjStep(nn.Module):
    def __init__(self, 
                 mj_model: mujoco.MjModel, 
                 mj_data: mujoco.MjData, 
                 n_steps: int = 1,
                 multithread: bool = False):
        """
        PyTorch nn.Module wrapper for differentiable MuJoCo step.
        
        Args:
            mj_model: mujoco.MjModel (Python object)
            mj_data: mujoco.MjData (Python object)
            n_steps: int, number of steps per forward call
            multithread: bool, use multithreading if True. default is False
        """
        super().__init__()
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.n_steps = n_steps
        self.multithread = multithread

    def forward(self, state: torch.Tensor, ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Call the custom autograd function with model and data pointers
        next_state, A, B = torchmj.MjStep.apply(
            state, ctrl, self.n_steps, self.mj_model._address, self.mj_data._address, self.multithread
        )

        return next_state
    
class PyMjStep(nn.Module):
    def __init__(self, 
                 mj_model: mujoco.MjModel, 
                 mj_data: mujoco.MjData, 
                 n_steps: int = 1):
        """
        PyTorch nn.Module wrapper for differentiable MuJoCo step.
        
        Args:
            mj_model: mujoco.MjModel (Python object)
            mj_data: mujoco.MjData (Python object)
            n_steps: int, number of steps per forward call
        """
        super().__init__()
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.n_steps = n_steps

    def forward(self, state: torch.Tensor, ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Call the custom autograd function with model and data
        next_state, _, _ = pyMjStep.apply(
            state, ctrl, self.n_steps, self.mj_model, self.mj_data
        )
    
        return next_state
