from torch.utils.cpp_extension import load

mjmod = load(
    name='mjmod',
    sources=['src/mjstep_module.cpp',
             'src/utils.cpp'],
    extra_include_paths=['/home/mukundan/opt/mujoco/mujoco-3.3.1/include'],
    extra_ldflags=[
        '-L/home/mukundan/opt/mujoco/mujoco-3.3.1/lib', 
        '-lmujoco'
    ],
    verbose=True,
)

import torch
import numpy as np
import mujoco

from python.autograd_mujoco import MjStep as pyMjStep
from python.utils import *

# device='cpu'
device='cuda'

if __name__=='__main__':
    mj_model=mujoco.MjModel.from_xml_path('assets/smpl_with_sensors.xml')
    # mj_model=mujoco.MjModel.from_xml_path('assets/half_cheetah.xml')
    data = mujoco.MjData(mj_model)

    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)

    qpos = torch.tensor(data.qpos, dtype=torch.float64, device=device).requires_grad_()
    if has_free_joint(mj_model): qpos = qpos_to_axis_angle(qpos)
    qvel = torch.tensor(data.qvel, dtype=torch.float64, device=device)
    act = torch.tensor(data.act, dtype=torch.float64, device=device) if mj_model.nu > 0 else torch.tensor([], dtype=torch.float64, device=device)
    sensordata = torch.tensor(data.sensordata, dtype=torch.float64, device=device)

    state = torch.cat([qpos, qvel, act, sensordata]).unsqueeze(0)

    ctrl=torch.zeros((1, mj_model.nu), dtype=torch.float64, device=device)

    # Example: pass int pointers for model and data (replace 0 with actual pointers later)
    out = mjmod.MjStep.apply(state, ctrl, 5, mj_model._address, data._address)
    print([o.shape for o in out])  # Should print 3 shapes

    next_state, A, B = pyMjStep.apply(state, ctrl, 5, mj_model, data)

    import pdb; pdb.set_trace()

    next_state_diff=out[0]-next_state
    A_diff=out[1]-A
    B_diff=out[2]-B


    import pdb; pdb.set_trace()






