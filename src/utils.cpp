#include "utils.hpp"

const double PI = 3.14159265358979323846; // Constant pi, self defined so you do not need cmath


/**
 * Converts axis-angle rotations to quaternions.
 *
 * @param axis_angle  Tensor of shape (..., 3), each vector is an axis-angle (axis * angle in radians).
 * @return            Tensor of shape (..., 4), corresponding quaternions (w, x, y, z), real part first.
 *
 * Notes:
 *  - Input and output are compatible with MuJoCo and PyTorch conventions.
 *  - Handles small angles safely with epsilon clamping.
 */

torch::Tensor axis_angle_to_quaternion(const torch::Tensor& axis_angle) {
    TORCH_CHECK(axis_angle.size(-1) == 3, "axis_angle must have 3 elements in last dimension");

    auto angles = axis_angle.norm(2, -1, true);
    angles = torch::clamp(angles, 1e-8);
    const double eps = std::numeric_limits<float>::epsilon();
    angles = torch::max(angles, torch::full_like(angles, eps));

    // Sinc in PyTorch is normalized: sinc(x) = sin(pi*x)/(pi*x)
    // Our use is for sin(x)/x, so adjust the argument: sinc(x/pi) = sin(x)/x
    // But torch::sinc input is in units of pi, so:
    //      torch::sinc(x / pi) = sin(x) / x
    // For half-angle:
    //      sin(angles/2) / angles = 0.5 * sinc(angles/2 / pi)
    auto sin_half_angles_over_angles = 0.5 * torch::sinc(angles * 0.5 / PI);

    // Form the quaternion: (cos(theta/2), axis * sin(theta/2)/theta)
    auto quat = torch::cat(
        {torch::cos(angles * 0.5), axis_angle * sin_half_angles_over_angles},
        -1
    );
    return quat;
}

/**
 * Converts the root joint's axis-angle rotation in qpos to a quaternion, 
 * and returns a new qpos with (x, y, z, qw, qx, qy, qz, ...) layout.
 *
 * @param qpos Input tensor of shape (..., N), N >= 6. 
 *        First 3 are translation, next 3 are axis-angle, rest are other joints.
 * @return     Tensor of shape (..., N + 1), with axis-angle replaced by quaternion.
 */

torch::Tensor axis_angle_root_to_quaternion(const torch::Tensor& qpos) {
    // Input: (..., N) where N >= 6 (first 3 = translation, next 3 = axis-angle)
    TORCH_CHECK(qpos.size(-1) >= 6, "qpos must have at least 6 elements in last dimension");

    auto root_trans = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto root_axis_angle = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});
    auto rest = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(6, torch::indexing::None)});

    // Convert root axis angle representation to quaternion
    auto root_quat = axis_angle_to_quaternion(root_axis_angle);

    // Concatenate along the last dimension
    return torch::cat(std::vector<at::Tensor>{root_trans, root_quat, rest}, -1);
}

/**
 * Converts unit quaternions to axis-angle representation.
 *
 * @param quat  Input tensor (..., 4), where last dimension is (w, x, y, z).
 * @param eps   Small epsilon to avoid division by zero (default: 1e-8).
 * @return      Tensor (..., 3), axis-angle vectors.
 */

torch::Tensor quaternion_to_axis_angle(const torch::Tensor& quat, double eps) {
    TORCH_CHECK(quat.size(-1) == 4, "quaternion must have 4 elements in last dimension");

    // Normalize for safety
    auto w   = quat.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)});   // (..., 1)
    auto xyz = quat.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});   // (..., 3)

    auto norm_xyz = xyz.norm(2, -1, true);    // (..., 1)
    auto angle = 2 * torch::atan2(norm_xyz, w); // (..., 1)

    // Prevent division by zero
    auto small = norm_xyz.lt(1e-8);  // ByteTensor (..., 1)
    auto axis = torch::where(small, torch::zeros_like(xyz), xyz / (norm_xyz + eps)); // (..., 3)

    return axis * angle;
}

/**
 * Converts MuJoCo-style qpos with root quaternion (qw, qx, qy, qz) to 
 * axis-angle root configuration.
 *
 * @param qpos  Input tensor (..., nq), first 3 entries are root translation,
 *              next 4 are quaternion (w, x, y, z), remaining are joints.
 * @param eps   Small value for numerical stability in quaternion-to-axis-angle.
 * @return      Tensor (..., nq - 1), with root represented as (x, y, z, axis_angle[3]), rest unchanged.
 *
 * Example:
 *     Input:  [x, y, z, qw, qx, qy, qz, ...]
 *     Output: [x, y, z, ax, ay, az, ...]
 */

torch::Tensor qpos_to_axis_angle(const torch::Tensor& qpos, double eps) {
    TORCH_CHECK(qpos.size(-1) >= 7, "qpos must have at least 7 elements in last dimension");

    auto root_translation = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto root_quat = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(3, 7)});
    auto root_axis_angle = quaternion_to_axis_angle(root_quat, eps);

    auto remaining = qpos.index({torch::indexing::Slice(), torch::indexing::Slice(7, torch::indexing::None)});

    return torch::cat(std::vector<at::Tensor>{root_translation, root_axis_angle, remaining}, -1);
}

/**
 * Loads a flat state tensor into MuJoCo's mjData fields.
 *
 * @param m          Pointer to MuJoCo mjModel struct.
 * @param d          Pointer to MuJoCo mjData struct.
 * @param state_row  Tensor (1D, double) with state values. 
 *                   Layout: [qpos (nq), qvel (nv), [act (na)], ...]
 *
 * Typical order for humanoids:
 *   - qpos: generalized positions (size nq)
 *   - qvel: generalized velocities (size nv)
 *   - act:  actuator states (optional, size na)
 *
 * This function fills d->qpos, d->qvel, and (if present) d->act from the tensor.
 */

void set_state_from_tensor(mjModel* m, mjData* d, const torch::Tensor& state_row) {
    TORCH_CHECK(state_row.numel() == m->nq+m->nv+m->na, 
                "state_row must have nq+nv+na elements");
    int off = 0;
    auto state_acc = state_row.accessor<double, 1>();
    for (int i = 0; i < m->nq; ++i)   d->qpos[i] = state_acc[off++];
    for (int i = 0; i < m->nv; ++i)   d->qvel[i] = state_acc[off++];
    if (m->na > 0)
      for (int i = 0; i < m->na; ++i) d->act[i]  = state_acc[off++];
}

/**
 * Loads a control tensor into MuJoCo's mjData struct.
 *
 * @param m         Pointer to MuJoCo mjModel struct.
 * @param d         Pointer to MuJoCo mjData struct.
 * @param ctrl_row  Tensor (1D, double) of length nu (number of controls).
 *
 * This function copies the values from the input tensor to d->ctrl.
 */

void set_ctrl_from_tensor(mjModel* m, mjData* d, const torch::Tensor& ctrl_row) {
    TORCH_CHECK(ctrl_row.numel() == m->nu, 
                "ctrl_row must have nu elements");
    auto ctrl_acc = ctrl_row.accessor<double, 1>();
    for (int i = 0; i < m->nu; ++i)
        d->ctrl[i] = ctrl_acc[i];
}

/**
 * Loads a sensordata tensor into MuJoCo's mjData struct.
 *
 * @param m                Pointer to MuJoCo mjModel struct.
 * @param d                Pointer to MuJoCo mjData struct.
 * @param sensordata_row   Tensor (1D, double) of length nsensordata.
 *
 * Copies values from sensordata_row to d->sensordata.
 */

void set_sensordata_from_tensor(mjModel* m, mjData* d, const torch::Tensor& sensordata_row) {
    TORCH_CHECK(sensordata_row.numel() == m->nsensordata, 
                "sensordata_row must have nsensordata elements");
    auto sens_acc = sensordata_row.accessor<double, 1>();
    for (int i = 0; i < m->nsensordata; ++i) {
        d->sensordata[i] = sens_acc[i];
    }
}

/**
 * Returns the MuJoCo state as a 1D torch tensor: [qpos, qvel, act].
 *
 * @param m   Pointer to mjModel.
 * @param d   Pointer to mjData.
 * @return    torch::Tensor of shape [nq + nv + na], dtype double.
 *
 * Note:
 *   - Always calls .clone() to ensure tensor owns its data
 *     (since std::vector will go out of scope).
 */

torch::Tensor get_state_from_mujoco(mjModel* m, mjData* d) {
    std::vector<double> out;
    out.insert(out.end(), d->qpos, d->qpos + m->nq);
    out.insert(out.end(), d->qvel, d->qvel + m->nv);
    if (m->na > 0)
        out.insert(out.end(), d->act, d->act + m->na);
    return torch::from_blob(out.data(), {(int)out.size()}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
}

/**
 * Returns the MuJoCo sensordata as a 1D torch tensor.
 *
 * @param m   Pointer to mjModel.
 * @param d   Pointer to mjData.
 * @return    torch::Tensor of shape [nsensordata], dtype double.
 *
 * Note:
 *   - .clone() is crucial: from_blob does NOT copy, so tensor will own its own memory.
 */

torch::Tensor get_sensor_from_mujoco(mjModel* m, mjData* d) {
    return torch::from_blob(d->sensordata, {m->nsensordata}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
}

/**
 * Runs a batched MuJoCo rollout using torch tensors.
 *
 * @param m         Pointer to mjModel.
 * @param d         Pointer to mjData (reset at each batch).
 * @param state     Tensor [B, N] with B batches, N state dim (qpos+qvel+...).
 * @param ctrl      Tensor [B, n_steps, U] with controls for each batch and step.
 *
 * @return (states, sensordatas):
 *   - states:      [B, n_steps, N] (all states at all steps for all batches)
 *   - sensordatas: [B, n_steps, nsensordata] (all sensors at all steps for all batches)
 *
 * Notes:
 *   - Each batch uses the same mjData (d); not thread-safe for parallel execution.
 *   - Assumes state is [qpos, qvel, (act)] matching set_state_from_tensor.
 *   - If act is present, set_state_from_tensor fills it.
 *   - Is the basis for the rollout loop used in the main module, left here for redundancy/prototyping purposes
 *   - Unused otherwise
 */

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

    auto states = torch::zeros({B, n_steps, N}, options);
    auto sensordatas = torch::zeros({B, n_steps, m->nsensordata}, options);

    for (int b = 0; b < B; ++b) {
        set_state_from_tensor(m, d, state[b]);
        for (int t = 0; t < n_steps; ++t) {
            set_ctrl_from_tensor(m, d, ctrl[b][t]);
            mj_step(m, d);
            states[b][t].copy_(get_state_from_mujoco(m, d));
            sensordatas[b][t].copy_(get_sensor_from_mujoco(m, d));
        }
    }
    return {states, sensordatas};
}

/**
 * Checks if the MuJoCo model has a free joint as its first joint.
 *
 * @param m  Pointer to mjModel.
 * @return   true if the first joint is type mjJNT_FREE, false otherwise.
 *
 * Note:
 *   - Assumes that the first joint (index 0) is the root. 
 *   - mjJNT_FREE means 6-DoF floating joint (for e.g. humanoid base).
 *   - For multi-body models, this is the standard way to check for floating bases.
 *   - Extend if your model has multiple free joints, you may need to modify other functions as well
 */

bool has_free_joint(const mjModel* m) {
    if (m->njnt > 0) {
        return m->jnt_type[0] == mjJNT_FREE;
    }
    return false;
}

/**
 * Checks if the MuJoCo model has any sensors.
 *
 * @param m  Pointer to mjModel.
 * @return   true if nsensor > 0, false otherwise.
 *
 * Typical use: controls whether to expect sensor data in rollout/state vectors.
 */

bool has_sensors(const mjModel* m) {
    return m->nsensor > 0;
}
