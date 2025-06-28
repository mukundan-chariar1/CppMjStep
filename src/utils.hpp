#pragma once
#include <torch/extension.h>
#include <mujoco/mujoco.h>

// All functions should work with torch::Tensor, just like in Python

// (..., 3) axis-angle → (..., 4) quaternion
torch::Tensor axis_angle_to_quaternion(const torch::Tensor& axis_angle);

// (..., N) qpos with axis-angle root → (..., N+1) qpos with quaternion root
torch::Tensor axis_angle_root_to_quaternion(const torch::Tensor& qpos);

// (..., 4) quaternion → (..., 3) axis-angle
torch::Tensor quaternion_to_axis_angle(const torch::Tensor& quat, double eps=1e-8);

// (..., nq) qpos with root quaternion → (..., 75) qpos with root axis-angle
torch::Tensor qpos_to_axis_angle(const torch::Tensor& qpos, double eps=1e-8);

std::tuple<torch::Tensor, torch::Tensor>
mj_batch_rollout(
    mjModel* m, mjData* d,
    const torch::Tensor& state,     // [B, N]
    const torch::Tensor& ctrl       // [B, n_steps, U]
);

void set_state_from_tensor(mjModel* m, mjData* d, const torch::Tensor& state_row);
void set_ctrl_from_tensor(mjModel* m, mjData* d, const torch::Tensor& ctrl_row);
void set_sensordata_from_tensor(mjModel* m, mjData* d, const torch::Tensor& sensordata_row);

bool has_free_joint(const mjModel* m);
bool has_sensors(const mjModel* m);


