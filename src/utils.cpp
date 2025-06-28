#include "utils.hpp"

const double PI = 3.14159265358979323846;

torch::Tensor axis_angle_to_quaternion(const torch::Tensor& axis_angle) {
    // (..., 3) -> (..., 4)
    auto angles = axis_angle.norm(2, -1, true);         // (..., 1)
    angles = torch::clamp(angles, 1e-8);                // avoid zero norm
    const double eps = std::numeric_limits<float>::epsilon();
    angles = torch::max(angles, torch::full_like(angles, eps));

    // 0.5 * sinc(angles * 0.5 / pi)
    // sinc(x) in torch is normalized: sinc(x) = sin(pi*x)/(pi*x)
    auto sin_half_angles_over_angles = 0.5 * torch::sinc(angles * 0.5 / PI);

    auto quat = torch::cat(
        {torch::cos(angles * 0.5), axis_angle * sin_half_angles_over_angles},
        -1
    ); // (..., 4)
    return quat;
}

torch::Tensor axis_angle_root_to_quaternion(const torch::Tensor& qpos) {
    // Input: (..., N) where N >= 6 (first 3 = translation, next 3 = axis-angle)
    TORCH_CHECK(qpos.size(-1) >= 6, "qpos must have at least 6 elements in last dimension");

    auto root_trans = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto root_axis_angle = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});
    auto rest = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(6, torch::indexing::None)});

    auto root_quat = axis_angle_to_quaternion(root_axis_angle);

    return torch::cat(std::vector<at::Tensor>{root_trans, root_quat, rest}, -1);
}

torch::Tensor quaternion_to_axis_angle(const torch::Tensor& quat, double eps) {
    // Ensure quaternions is (..., 4)
    TORCH_CHECK(quat.size(-1) == 4, "quaternion must have 4 elements in last dimension");

    // Normalize for safety
    auto w   = quat.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)});   // (..., 1)
    auto xyz = quat.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});   // (..., 3)

    auto norm_xyz = xyz.norm(2, -1, true);    // (..., 1)
    auto angle = 2 * torch::atan2(norm_xyz, w); // (..., 1)

    // Prevent division by zero
    auto small = norm_xyz.lt(1e-8);  // ByteTensor (..., 1)
    auto axis = torch::where(small, torch::zeros_like(xyz), xyz / (norm_xyz + 1e-8)); // (..., 3)

    return axis * angle;
}

torch::Tensor qpos_to_axis_angle(const torch::Tensor& qpos, double eps) {
    // Input: (..., nq), with at least 7 elements in last dim (x, y, z, qw, qx, qy, qz)
    TORCH_CHECK(qpos.size(-1) >= 7, "qpos must have at least 7 elements in last dimension");

    auto root_translation = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto root_quat = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(3, 7)});
    auto root_axis_angle = quaternion_to_axis_angle(root_quat, eps);

    auto remaining = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(7, torch::indexing::None)});

    return torch::cat(std::vector<at::Tensor>{root_translation, root_axis_angle, remaining}, -1);
}

// Helper to copy state tensor to mjData (adapt as needed)
void set_state_from_tensor(mjModel* m, mjData* d, const torch::Tensor& state_row) {
    // Assume: state_row [N], with qpos, qvel, (optionally act, sensordata)
    // For typical humanoid: [qpos (nq), qvel (nv), act (na)]
    int off = 0;
    auto state_acc = state_row.accessor<double, 1>();
    for (int i = 0; i < m->nq; ++i)   d->qpos[i] = state_acc[off++];
    for (int i = 0; i < m->nv; ++i)   d->qvel[i] = state_acc[off++];
    if (m->na > 0)
      for (int i = 0; i < m->na; ++i) d->act[i]  = state_acc[off++];
    // Add others if needed
}

void set_ctrl_from_tensor(mjModel* m, mjData* d, const torch::Tensor& ctrl_row) {
    auto ctrl_acc = ctrl_row.accessor<double, 1>();
    for (int i = 0; i < m->nu; ++i)
        d->ctrl[i] = ctrl_acc[i];
}

void set_sensordata_from_tensor(mjModel* m, mjData* d, const torch::Tensor& sensordata_row) {
    TORCH_CHECK(sensordata_row.numel() == m->nsensordata, 
                "sensordata_row must have nsensordata elements");
    auto sens_acc = sensordata_row.accessor<double, 1>();
    for (int i = 0; i < m->nsensordata; ++i) {
        d->sensordata[i] = sens_acc[i];
    }
}

torch::Tensor get_state_from_mujoco(mjModel* m, mjData* d) {
    // Returns a tensor [nq+nv+na]
    std::vector<double> out;
    out.insert(out.end(), d->qpos, d->qpos + m->nq);
    out.insert(out.end(), d->qvel, d->qvel + m->nv);
    if (m->na > 0)
        out.insert(out.end(), d->act, d->act + m->na);
    // If you want to add sensordata, add here.
    return torch::from_blob(out.data(), {(int)out.size()}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
}

torch::Tensor get_sensor_from_mujoco(mjModel* m, mjData* d) {
    return torch::from_blob(d->sensordata, {m->nsensordata}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
}

std::tuple<torch::Tensor, torch::Tensor>
mj_batch_rollout(
    mjModel* m, mjData* d,
    const torch::Tensor& state,     // [B, N]
    const torch::Tensor& ctrl       // [B, n_steps, U]
) {
    int B = state.size(0);
    int N = state.size(1);
    int n_steps = ctrl.size(1);
    int U = ctrl.size(2);

    auto options = state.options();

    // Prepare output
    auto states = torch::zeros({B, n_steps, N}, options);
    auto sensordatas = torch::zeros({B, n_steps, m->nsensordata}, options);

    for (int b = 0; b < B; ++b) {
        // Set initial state for this batch
        set_state_from_tensor(m, d, state[b]);
        for (int t = 0; t < n_steps; ++t) {
            set_ctrl_from_tensor(m, d, ctrl[b][t]);
            mj_step(m, d);
            // Save state
            states[b][t].copy_(get_state_from_mujoco(m, d));
            // Save sensors
            sensordatas[b][t].copy_(get_sensor_from_mujoco(m, d));
        }
    }
    return {states, sensordatas};
}

// Return true if the model has a free joint as its first joint
bool has_free_joint(const mjModel* m) {
    if (m->njnt > 0) {
        return m->jnt_type[0] == mjJNT_FREE;
    }
    return false;
}

bool has_sensors(const mjModel* m) {
    return m->nsensor > 0;
}
