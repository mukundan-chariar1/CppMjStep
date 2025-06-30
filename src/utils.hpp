#pragma once
#include <torch/extension.h>
#include <mujoco/mujoco.h>

torch::Tensor axis_angle_to_quaternion(const torch::Tensor& axis_angle);
torch::Tensor axis_angle_root_to_quaternion(const torch::Tensor& qpos);
torch::Tensor quaternion_to_axis_angle(const torch::Tensor& quat, double eps=1e-8);
torch::Tensor qpos_to_axis_angle(const torch::Tensor& qpos, double eps=1e-8);

std::tuple<torch::Tensor, torch::Tensor>
mj_batch_rollout(
    mjModel* m, mjData* d,
    const torch::Tensor& state,
    const torch::Tensor& ctrl
);

void set_state_from_tensor(mjModel* m, mjData* d, const torch::Tensor& state_row);
void set_ctrl_from_tensor(mjModel* m, mjData* d, const torch::Tensor& ctrl_row);
void set_sensordata_from_tensor(mjModel* m, mjData* d, const torch::Tensor& sensordata_row);

bool has_free_joint(const mjModel* m);
bool has_sensors(const mjModel* m);

torch::Tensor get_state_from_mujoco(mjModel* m, mjData* d);
torch::Tensor get_sensor_from_mujoco(mjModel* m, mjData* d);
