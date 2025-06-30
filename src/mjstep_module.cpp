#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include "utils.hpp"

namespace py = pybind11;

struct MjStep : public torch::autograd::Function<MjStep> {
    /**
     * Static forward method for custom MuJoCo autograd Function.
     *
     * Implements the differentiable forward MuJoCo rollout, with optional Jacobian computation.
     *
     * @param ctx           Autograd context to save tensors/objects for the backward pass.
     * @param state         [B, nq + nv + na + nsensordata] State tensor for each batch (batch size B).
     * @param ctrl          [B, nu] Control input tensor for each batch.
     * @param n_steps       Number of simulation steps to roll forward.
     * @param mj_model_ptr  Opaque pointer (uintptr_t) to MuJoCo mjModel struct.
     * @param mj_data_ptr   Opaque pointer (uintptr_t) to MuJoCo mjData struct.
     *
     * @return torch::autograd::tensor_list
     *         - next_state: [B, ...] Tensor of next state(s) after simulation.
     *         - dydx      : [B, ...] State-state Jacobian (optional, or empty).
     *         - dydu      : [B, ...] State-control Jacobian (optional, or empty).
     *
     * Notes:
     *   - Tensors must have appropriate shape, device, dtype, and be contiguous.
     *   - `ctx` should be used to save any tensors/objects needed for backward().
     *   - mjModel and mjData pointers must be valid and owned/managed on the Python side.
     */

    static torch::autograd::tensor_list
    forward(torch::autograd::AutogradContext *ctx,
            torch::Tensor state,
            torch::Tensor ctrl,
            int64_t n_steps,
            uintptr_t mj_model_ptr,
            uintptr_t mj_data_ptr) 
    {
        // Unpack pointers to MuJoCo
        mjModel* m = reinterpret_cast<mjModel*>(mj_model_ptr);
        mjData* d = reinterpret_cast<mjData*>(mj_data_ptr);

        // Batch and dimension (packed state) info
        int B = state.size(0);
        int D = state.size(1);

        // Model dimension parameters from mjModel:
        //
        // nq           : Number of position variables (generalized coordinates).
        // nv           : Number of velocity variables (generalized velocities).
        // na           : Number of actuator activation variables (for muscle/actuator models).
        // nu           : Number of actuator control variables (inputs).
        // nsensordata  : Number of sensor data outputs per time step.
        int nq = m->nq;
        int nv = m->nv;
        int na = m->na;
        int nu = m->nu;
        int nsensordata = m->nsensordata;

        // Derived model dimensions:
        //
        // nA : State-action Jacobian size (A matrix), computed as (2 * nv + na).
        //      - 2 * nv: Double the number of velocities (typically position and velocity).
        //      - na   : Number of actuator activation variables.
        //      Total: State dimension for linearization.
        //
        // nU : Control input dimension (B matrix), same as nu.
        //
        // nC : Output (sensor + actuator) dimension for C/D matrices, computed as (nsensordata + na).
        //      - nsensordata: Number of sensor data outputs.
        //      - na        : Number of actuator activations.
        int nA = 2 * m->nv + m->na;
        int nU = m->nu;
        int nC = m->nsensordata + m->na;

        int total_dim = nA + nC; // Total dimension of dydx tensor

        int N = m->nq+m->nv+m->na; //Total dimension of unpacked state

        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        auto dydx = torch::zeros({B, total_dim, total_dim}, options);
        auto dydu = torch::zeros({B, total_dim, nU}, options);

        // Optionally: check tensor types and shapes for robustness!
        TORCH_CHECK(state.dim() == 2, "state must be [B, D]");
        TORCH_CHECK(ctrl.dim() == 2, "ctrl must be [B, nu]");
        TORCH_CHECK(D >= nsensordata, "state does not have enough dims for sensordata");

        // Get device for outputs, save state options to cast back
        auto device = state.device();
        auto init_options = state.options();

        // Check if gradients needed
        bool compute_grads = state.requires_grad() || ctrl.requires_grad();
        bool sensor_flag = has_sensors(m);
        bool free_joint_flag = has_free_joint(m);

        // Make CPU, contiguous, float64 copies (MuJoCo needs double*, you might want to work in double)
        torch::Tensor state_cpu = state.to(torch::kCPU).contiguous().to(torch::kFloat64);
        torch::Tensor ctrl_cpu  = ctrl.to(torch::kCPU).contiguous().to(torch::kFloat64);

        // Preallocate sensordata and state variables
        torch::Tensor sensordata;
        torch::Tensor state_no_sensor = state_cpu;

        // If sensors are present, state has sensordata added to the end
        if (sensor_flag) {
            state_no_sensor = state_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(0, D - nsensordata)});
            sensordata = state_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(D - nsensordata, torch::indexing::None)});
        } else {
            sensordata = torch::empty(
                {state_cpu.size(0), 0},
                state_cpu.options()
            );
        }

        // If free joint present at root, we unpack axis angle to a quaternion in accordance with mujoco representation of free joints
        if (free_joint_flag) {
            state_no_sensor = axis_angle_root_to_quaternion(state_no_sensor);
        }

        // Preallocate states and sensordatas (plural).
        auto states = torch::zeros({B, n_steps+1, N}, options);
        auto sensordatas = torch::zeros({B, n_steps+1, m->nsensordata}, options);

        // Batch loop
        for (int b = 0; b < B; ++b) {
            // Set initial state for this batch
            set_state_from_tensor(m, d, state_no_sensor[b]);

            // Initialize states and sensordatas tensor with initial state and sensordata
            states[b][0].copy_(state_no_sensor[b]);
            sensordatas[b][0].copy_(sensordata[b]);

            torch::Tensor A = torch::eye(nA, options);                // [nA, nA]
            torch::Tensor B = torch::zeros({nA, nU}, options);        // [nA, nU]
            torch::Tensor C, D;                                       // Will be set later

            torch::Tensor _A = torch::zeros({nA, nA}, options);       // Temporary A tensor
            torch::Tensor _B = torch::zeros({nA, nU}, options);       // Temporary B tensor
            torch::Tensor _C = torch::zeros({nC, nA}, options);       // Temporary C tensor
            torch::Tensor _D = torch::zeros({nC, nU}, options);       // Temporary D tensor

            // Timestep loop
            for (int t = 0; t < n_steps; ++t) {
                set_state_and_ctrl(m, d, states[b][t], ctrl_cpu[b], sensordatas[b][t]);
                
                if (compute_grads){
                    m->opt.tolerance    = 0;          // Disable early termination
                    m->opt.ls_tolerance = 1e-18;      // Very low line search tolerance
                    m->opt.disableflags = 1 << 8;     // Disable solver warmstart (2^8 = 256)

                    // Most of the time spent is in the next command, if mjd_transitionFD is sped up, the code speeds up as well.
                    
                    mjd_transitionFD(m, d, 1e-8, 1, 
                        _A.data_ptr<double>(), _B.data_ptr<double>(), _C.data_ptr<double>(), _D.data_ptr<double>());

                    // Update matrices (matmul in torch)
                    A = torch::matmul(_A, A);             // [nA, nA]
                    B = torch::matmul(_A, B) + _B;        // [nA, nU]

                    m->opt.tolerance    = 1e-10;      // Re-enable early termination
                    m->opt.ls_tolerance = 1e-10;      // Re-set line search tolerance
                    m->opt.disableflags = 0;          // Re-enable solver warmstart
                }

                // Step the simulation
                mj_step(m, d);

                // Save state and sensors in the next index of the tensor
                states[b][t+1].copy_(get_state_from_mujoco(m, d));
                sensordatas[b][t+1].copy_(get_sensor_from_mujoco(m, d));
            }

            // Update matrices (matmul in torch) at the end of the loop to save some computation (not much)
            C = torch::matmul(_C, A);             // [nC, nA]
            D = torch::matmul(_C, B) + _D;        // [nC, nU]

            // Save the batch in dydx and dydu
            dydx[b].index_put_({torch::indexing::Slice(0, nA), torch::indexing::Slice(0, nA)}, A);
            dydx[b].index_put_({torch::indexing::Slice(nA, total_dim), torch::indexing::Slice(0, nA)}, C);
            dydu[b].copy_(torch::cat({B, D}, 0));
        }

        // Pull out the final state and tensordata
        auto next_state      = states.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});         // [B, ...]
        auto sensordatas_final = sensordatas.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});    // [B, ...]

        // If freejoint is present, pack quaternion into axis angle
        if (free_joint_flag) {
            next_state = qpos_to_axis_angle(next_state);
        }

        // Concatenate with sensordata along the last dimension
        next_state = torch::cat({next_state, sensordatas_final}, -1);

        // Cast to output dtype/device (from initial state)
        next_state = next_state.to(init_options);
        dydx = dydx.to(init_options);
        dydu = dydu.to(init_options);

        // If gradients are computed, save context for backward pass
        if (compute_grads) {
            ctx->save_for_backward({dydx, dydu});
        } 
        // Return as tuple
        return {next_state, dydx, dydu};
    }

    /**
     * Set MuJoCo state, control, and sensor data from torch tensors.
     *
     * @param m               Pointer to MuJoCo mjModel struct.
     * @param d               Pointer to MuJoCo mjData struct.
     * @param state_row       Tensor of state variables for a single batch element (e.g. [nq + nv + na]).
     * @param ctrl_row        Tensor of control variables for a single batch element (e.g. [nu]).
     * @param sensordata_row  Tensor of sensor data for a single batch element (e.g. [nsensordata]).
     *
     * This function copies the contents of the provided tensors into the MuJoCo data structure,
     * updating qpos, qvel, act, ctrl, and sensordata as appropriate.
     */

    static void set_state_and_ctrl(mjModel* m, mjData* d,
        const torch::Tensor& state_row,
        const torch::Tensor& ctrl_row,
        const torch::Tensor& sensordata_row)
    {
        set_state_from_tensor(m, d, state_row);
        set_ctrl_from_tensor(m, d, ctrl_row);
        set_sensordata_from_tensor(m, d, sensordata_row);
    }

    /**
     * Backward pass for custom MuJoCo autograd Function.
     *
     * Computes the gradients of the loss with respect to input state and control,
     * using the saved Jacobians from the forward pass.
     *
     * @param ctx           Autograd context containing saved variables from forward().
     * @param grad_outputs  List of gradients with respect to each forward output.
     *                      grad_outputs[0]: Gradient w.r.t. next_state, shape [B, D].
     *
     * @return tensor_list of gradients:
     *    - grad_state : [B, D]   Gradient w.r.t. input state.
     *    - grad_ctrl  : [B, U]   Gradient w.r.t. input control.
     *    - (None for n_steps, mj_model_ptr, mj_data_ptr)
     *
     * Notes:
     *    - The function uses saved dydx and dydu from forward().
     *    - Batch matrix multiply is used to propagate gradients efficiently.
     *    - Only gradients for state and control are returned; the rest are None.
     */

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // Unpack saved tensors from forward
        auto saved = ctx->get_saved_variables();
        torch::Tensor dydx = saved[0];
        torch::Tensor dydu = saved[1];

        // grad_outputs[0] is grad w.r.t. output (next_state), shape: [B, D]
        auto grad_output = grad_outputs[0];  // [B, D]

        torch::Tensor grad_state, grad_ctrl;

        // Check if state requires grad
        if (ctx->needs_input_grad(0)) {
            // grad_output: [B, D], dydx: [B, D, D]
            grad_state = torch::bmm(
                grad_output.unsqueeze(1), // [B, 1, D]
                dydx                       // [B, D, D]
            ).squeeze(1);                  // [B, D]
        } else {
            grad_state = torch::Tensor();  // None
        }

        // Check if ctrl requires grad
        if (ctx->needs_input_grad(1)) {
            // grad_output: [B, D], dydu: [B, D, U]
            grad_ctrl = torch::bmm(
                grad_output.unsqueeze(1), // [B, 1, D]
                dydu                      // [B, D, U]
            ).squeeze(1);                 // [B, U]
        } else {
            grad_ctrl = torch::Tensor();  // None
        }

        // Return gradients for: state, ctrl, n_steps, mj_model_ptr, mj_data_ptr
        // Only first two require gradients; the rest should be None.
        return {grad_state, grad_ctrl, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

/**
 * Python binding helper for MjStep::apply.
 *
 * Forwards arguments to the static PyTorch autograd Function MjStep::apply,
 * and converts the result to a Python tuple for returning to Python code.
 *
 * @param state         [B, D]  Input state tensor (batch, state_dim).
 * @param ctrl          [B, U]  Input control tensor (batch, control_dim).
 * @param n_steps       Number of rollout steps.
 * @param mj_model_ptr  Opaque pointer (uintptr_t) to MuJoCo mjModel struct.
 * @param mj_data_ptr   Opaque pointer (uintptr_t) to MuJoCo mjData struct.
 *
 * @return py::object   A Python tuple of (next_state, dydx, dydu) tensors.
 *
 * Notes:
 *   - This helper is exposed to Python via pybind11 as a function.
 *   - Tensors must be on correct device/dtype and match expected shapes.
 *   - mjModel and mjData must be managed/lifetime-safe on Python side.
 */

py::object mjstep_apply(torch::Tensor state,
                        torch::Tensor ctrl,
                        int64_t n_steps,
                        uintptr_t mj_model_ptr,
                        uintptr_t mj_data_ptr)
{
    // Call the static apply method, which returns a tuple of tensors
    auto result = MjStep::apply(state, ctrl, n_steps, mj_model_ptr, mj_data_ptr);
    return py::cast(result);  // Converts std::tuple<Tensor, Tensor, Tensor> to Python tuple
}

/**
 * Pybind11 module definition for custom MuJoCo PyTorch extension.
 *
 * Exposes the MjStep autograd Function as a Python class, with a static
 * .apply() method for differentiable MuJoCo rollouts.
 *
 * Usage in Python:
 *     next_state, dydx, dydu = mjmod.MjStep.apply(state, ctrl, n_steps, mj_model_ptr, mj_data_ptr)
 *
 * Arguments:
 *     state         - Input state tensor, shape [B, D]
 *     ctrl          - Input control tensor, shape [B, U]
 *     n_steps       - Number of simulation steps to roll forward
 *     mj_model_ptr  - Pointer to MuJoCo mjModel (as int/ctypes address)
 *     mj_data_ptr   - Pointer to MuJoCo mjData (as int/ctypes address)
 *
 * Returns:
 *     next_state    - Output state tensor after simulation
 *     dydx          - Linearization Jacobian w.r.t. state
 *     dydu          - Linearization Jacobian w.r.t. control
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MjStep>(m, "MjStep")
    .def_static("apply", &mjstep_apply,
        py::arg("state"),
        py::arg("ctrl"),
        py::arg("n_steps"),
        py::arg("mj_model_ptr"),
        py::arg("mj_data_ptr"),
        R"pbdoc(
                Differentiable MuJoCo step.

                Static apply method for batched, differentiable MuJoCo simulation.

                Parameters
                ----------
                state : torch.Tensor
                    Input state tensor of shape [B, D], where B is batch size and D is state dimension (e.g., nq + nv + na [+ nsensordata]).
                ctrl : torch.Tensor
                    Control input tensor of shape [B, U], where U is the number of controls (nu).
                n_steps : int
                    Number of simulation steps to roll forward.
                mj_model_ptr : int
                    Integer pointer (address) to the MuJoCo mjModel structure.
                mj_data_ptr : int
                    Integer pointer (address) to the MuJoCo mjData structure.

                Returns
                -------
                next_state : torch.Tensor
                    Output tensor of next state(s) after simulation, shape [B, D].
                dydx : torch.Tensor or None
                    Jacobian of next_state with respect to input state, shape [B, D, D], or None if gradients are not computed.
                dydu : torch.Tensor or None
                    Jacobian of next_state with respect to control, shape [B, D, U], or None if gradients are not computed.

                Notes
                -----
                This method enables differentiable MuJoCo rollouts inside PyTorch computation graphs, with optional analytic Jacobians.
                All pointers (mj_model_ptr, mj_data_ptr) must remain valid during simulation.

                Example
                -------
                >>> next_state, dydx, dydu = MjStep.apply(state, ctrl, n_steps, mj_model_ptr, mj_data_ptr)
        )pbdoc"
    );
}
