# Installation Instructions

## 1. Requirements

* **Python 3.8+**
* **PyTorch** (tested with >= 1.13)
* **MuJoCo** 3.x (compiled and installed, see [MuJoCo install guide](https://mujoco.readthedocs.io/en/stable/))
* **setuptools**
* **C++17 compiler** (e.g. g++ >= 7)
* (Recommended) **virtual environment** (conda or venv)

## 2. MuJoCo Setup

1. **Download MuJoCo** and extract to a directory (e.g. `~/opt/mujoco/mujoco-3.3.1`).

2. **Set environment variables** *(optional but helpful)*:

   ```bash
   export MUJOCO_PY_MUJOCO_PATH=~/opt/mujoco/mujoco-3.3.1
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/opt/mujoco/mujoco-3.3.1/lib
   ```

## 3. Clone This Repository

```bash
git clone git@github.com:mukundan-chariar1/CppMjStep.git
cd CppMjStep
```
* **Note:** Update your `setup.py` or build scripts with the correct MuJoCo include and lib paths.

## 4. (Recommended) Create a Python Environment

```bash
conda create -n MjStep_env python=3.10
conda activate MjStep_env
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng
```

## 5. Install the Package

```bash
pip install -e .
```

* This will build the C++ extension and install your Python package in “editable” mode.

## 6. Test Your Installation

You can now import your module in Python:

```python
import torch
import mujoco as mj
from CppMjStep import MjStep, PyMjStep

# Initialize MuJoCo model and data
xml_path = 'path/to/your/model.xml'
mj_model = mj.MjModel.from_xml_path(filename=xml_path)
mj_data = mj.MjData(mj_model)

# Define MjStep layer
torch_wrapped_model = MjStep(mj_model, mj_data, n_steps=5)

# Define initial state and control input tensors
state = torch.rand(mj_model.nq + 
                    mj_model.nv + 
                    mj_model.na + 
                    mj_model.nsensordata, 
                    requires_grad=True)
ctrl = torch.rand(mj_model.nu, requires_grad=True)

# Compute next state
next_state = MjStep(state, ctrl)
```
---
## 7. Additional Resources

* [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/)
* [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

---

