import torch
import numpy as np
import mujoco

def has_free_joint(m):
    return m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

def has_sensors(m):
    return m.nsensor > 0

def quaternion_to_axis_angle(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert unit quaternions to axis-angle representation.

    Args:
        quaternions: Tensor of shape (..., 4), in (w, x, y, z) format.
                     Must be unit quaternions (normalized).

    Returns:
        Tensor of shape (..., 3), axis-angle vectors where direction is the rotation axis,
        and norm is the rotation angle in radians.
    """
    # Normalize for safety
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    w = quaternions[..., 0:1]         # shape (..., 1)
    vec = quaternions[..., 1:]        # shape (..., 3)

    sin_half_theta = vec.norm(dim=-1, keepdim=True)  # (..., 1)
    theta = 2 * torch.atan2(sin_half_theta, w)       # (..., 1)

    # Avoid division by zero
    scale = theta / (sin_half_theta + eps)
    axis_angle = vec * scale                         # (..., 3)

    return axis_angle

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    angles = angles.clamp(min=1e-8)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )

def qpos_to_axis_angle(qpos: torch.Tensor) -> torch.Tensor:
    """
    Converts MuJoCo-style qpos (with quaternion at root) to a (75,) axis-angle style configuration vector.

    Parameters
    ----------
    qpos : torch.Tensor
        Tensor of shape (..., nq), where nq >= 7 and the first 7 elements are (x, y, z, qw, qx, qy, qz).

    Returns
    -------
    q : torch.Tensor
        Tensor of shape (..., 75), where the first 6 elements are (x, y, z, axis_angle[3]) and the rest are unchanged.
    """
    root_translation = qpos[..., :3]
    root_quat = qpos[..., 3:7]  # (w, x, y, z)
    root_axis_angle = quaternion_to_axis_angle(root_quat)

    remaining = qpos[..., 7:]  # Should be shape (..., 69)
    return torch.cat([root_translation, root_axis_angle, remaining], dim=-1)

def axis_angle_root_to_quaternion(qpos: torch.Tensor) -> torch.Tensor:
    """
    Converts the root joint rotation in axis-angle (3,) format to a quaternion (4,)
    and returns a new qpos tensor with the first 3 + 4 values being (xyz translation + quaternion),
    followed by the rest of the original qpos vector (excluding the original axis-angle).

    Assumes qpos starts with:
        - 3 values for root translation
        - 3 values for root axis-angle rotation
        - followed by other joint rotations (e.g. 69-3-3 = 63 values)

    Parameters
    ----------
    qpos : torch.Tensor
        Tensor of shape (..., N) where N ≥ 6 and first 6 entries are root translation + root axis-angle.

    Returns
    -------
    torch.Tensor
        Updated qpos of shape (..., N + 1), where axis-angle is replaced by quaternion.
    """
    root_trans = qpos[..., :3]
    root_axis_angle = qpos[..., 3:6]
    rest = qpos[..., 6:]

    root_quat = axis_angle_to_quaternion(root_axis_angle)

    return torch.cat([root_trans, root_quat, rest], dim=-1)

def axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to quaternions (NumPy version).

    Args:
        axis_angle: (..., 3) array in axis-angle form.

    Returns:
        (..., 4) array of quaternions with real part first.
    """
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    angles = angles.clip(min=1e-8)
    eps = np.finfo(np.float32).eps
    angles = np.maximum(angles, eps)  # avoid divide-by-zero

    sin_half_angles_over_angles = 0.5 * np.sinc(angles * 0.5 / np.pi)
    quat = np.concatenate(
        [np.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quat

def axis_angle_root_to_quaternion_np(qpos: np.ndarray) -> np.ndarray:
    """
    Converts root axis-angle rotation in qpos to quaternion.

    Parameters
    ----------
    qpos : np.ndarray
        (..., N), where N ≥ 6. First 6 elements = [xyz, axis-angle].

    Returns
    -------
    np.ndarray
        (..., N+1), with axis-angle replaced by quaternion (4D).
    """
    root_trans = qpos[..., :3]
    root_axis_angle = qpos[..., 3:6]
    rest = qpos[..., 6:]

    root_quat = axis_angle_to_quaternion_np(root_axis_angle)
    return np.concatenate([root_trans, root_quat, rest], axis=-1)

def quaternion_to_axis_angle_np(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to axis-angle representation.
    
    Parameters
    ----------
    quat : np.ndarray
        Array of shape (..., 4), with w as the real component.

    Returns
    -------
    axis_angle : np.ndarray
        Array of shape (..., 3), axis-angle representation.
    """
    w, xyz = quat[..., 0:1], quat[..., 1:]
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm_xyz, w)

    # Prevent division by zero
    small = norm_xyz < 1e-8
    axis = np.where(small, np.zeros_like(xyz), xyz / norm_xyz)

    return axis * angle


def qpos_to_axis_angle_np(qpos: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo-style qpos with quaternion root to axis-angle representation.

    Parameters
    ----------
    qpos : np.ndarray
        Shape (..., nq), with first 7 values as (x, y, z, qw, qx, qy, qz).

    Returns
    -------
    q : np.ndarray
        Shape (..., 75), with (x, y, z, axis_angle[3], rest).
    """
    root_translation = qpos[..., :3]
    root_quat = qpos[..., 3:7]
    root_axis_angle = quaternion_to_axis_angle_np(root_quat)
    rest = qpos[..., 7:]
    
    return np.concatenate([root_translation, root_axis_angle, rest], axis=-1)