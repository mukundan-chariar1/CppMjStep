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
    // Forward: static function!
    static torch::autograd::tensor_list
    forward(torch::autograd::AutogradContext *ctx,
            torch::Tensor state,
            torch::Tensor ctrl,
            int64_t n_steps,
            uintptr_t mj_model_ptr,
            uintptr_t mj_data_ptr) 
    {
        // C++ (for a custom autograd::Function or just for your function)

        // Unpack pointers to MuJoCo
        mjModel* m = reinterpret_cast<mjModel*>(mj_model_ptr);
        mjData* d = reinterpret_cast<mjData*>(mj_data_ptr);

        // Batch and dimension info
        int B = state.size(0);
        int D = state.size(1);
        int nsensordata = m->nsensordata;

        // Optionally: check tensor types and shapes for robustness!
        TORCH_CHECK(state.dim() == 2, "state must be [B, D]");
        TORCH_CHECK(ctrl.dim() == 2, "ctrl must be [B, nu]");
        TORCH_CHECK(D >= nsensordata, "state does not have enough dims for sensordata");

        // Get device for outputs
        auto device = state.device();

        // Check if gradients needed
        bool compute_grads = state.requires_grad() || ctrl.requires_grad();
        bool sensor_flag = has_sensors(m);
        bool free_joint_flag = has_free_joint(m);

        // Make CPU, contiguous, float64 copies (MuJoCo needs double*, you might want to work in double)
        torch::Tensor state_cpu = state.to(torch::kCPU).contiguous().to(torch::kFloat64);
        torch::Tensor ctrl_cpu  = ctrl.to(torch::kCPU).contiguous().to(torch::kFloat64);

        // Split out the sensordata portion (last nsensordata dims)
        int state_feat = D - nsensordata;  // everything except sensordata

        torch::Tensor sensordata;
        torch::Tensor state_no_sensor = state_cpu;

        if (sensor_flag) {
            // state: [B, D]
            state_no_sensor = state_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(0, D - nsensordata)});
            sensordata = state_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(D - nsensordata, torch::indexing::None)});
        } else {
            sensordata = torch::empty(
                {state_cpu.size(0), 0},
                state_cpu.options()
            );
        }

        if (free_joint_flag) {
            state_no_sensor = axis_angle_root_to_quaternion(state_no_sensor);
        }

        torch::Tensor ctrl_unsqueezed = ctrl_cpu.unsqueeze(1);         // [B, 1, U]
        torch::Tensor ctrl_repeated = ctrl_unsqueezed.repeat({1, n_steps, 1}); // [B, n_steps, U]

        int mj_state_size = mj_stateSize(m, mjtState::mjSTATE_FULLPHYSICS);

        auto [states, sensordatas] = mj_batch_rollout(m, d, state_no_sensor, ctrl_repeated);

        int nq = m->nq;
        int nv = m->nv;
        int na = m->na;
        int nu = m->nu;

        int batch_size = state_cpu.size(0);
        int nA = 2 * m->nv + m->na;
        int nU = m->nu;
        int nC = m->nsensordata + m->na;

        // [B, ..., total_state_dim] → [B, n_steps, nq + nv + na]
        states = states.reshape({-1, n_steps, nq + nv + na});
        ctrl_repeated = ctrl_repeated.reshape({-1, n_steps, nu});
        if (sensor_flag) {
            sensordatas = sensordatas.reshape({-1, n_steps, m->nsensordata});
        }

        std::vector<torch::Tensor> dydx_vec, dydu_vec;

        if (compute_grads) {
            // Expand state: [B, D] -> [B, 1, D]
            auto state_unsqueezed = state_no_sensor.unsqueeze(1);

            // states: [B, n_steps, D] -> [B, n_steps-1, D]
            auto states_sliced = states.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1), torch::indexing::Slice()});

            // Concatenate: [B, 1, D] + [B, n_steps-1, D] -> [B, n_steps, D]
            auto _states = torch::cat({state_unsqueezed, states_sliced}, 1);

            // Same for sensordata
            auto sensordata_unsqueezed = sensordata.unsqueeze(1); // [B, 1, S]
            auto sensordatas_sliced = sensordatas.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1), torch::indexing::Slice()});
            auto _sensordatas = torch::cat({sensordata_unsqueezed, sensordatas_sliced}, 1);

            m->opt.tolerance    = 0;          // Disable early termination
            m->opt.ls_tolerance = 1e-18;      // Very low line search tolerance
            m->opt.disableflags = 1 << 8;     // Disable solver warmstart (2^8 = 256)

            for (int batch = 0; batch < batch_size; ++batch) {
                auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
                // Or reuse options from your forward tensors

                auto state_batch      = _states.index({batch});      // [n_steps, ...]
                auto ctrl_batch       = ctrl_repeated.index({batch});         // [n_steps, ...]
                auto sensordata_batch = _sensordatas.index({batch}); // [n_steps, ...]

                torch::Tensor A = torch::eye(nA, options);                // [nA, nA]
                torch::Tensor B = torch::zeros({nA, nU}, options);        // [nA, nU]
                torch::Tensor C, D;                                       // Will be set later

                for (int step = 0; step < n_steps; ++step) {
                    // Unpack per-step state, ctrl, sensordata as 1D tensors (already torch::Tensor)
                    auto _state      = state_batch[step];      // [nA]
                    auto _ctrl       = ctrl_batch[step];       // [nU]
                    auto _sensordata = sensordata_batch[step]; // [nC]

                    // Reset and set MuJoCo
                    mj_resetData(m, d);
                    set_state_and_ctrl(m, d, _state, _ctrl, _sensordata); // You must implement this!

                    // Allocate all-zero tensors for MuJoCo FD output
                    torch::Tensor _A = torch::zeros({nA, nA}, options);
                    torch::Tensor _B = torch::zeros({nA, nU}, options);
                    torch::Tensor _C = torch::zeros({nC, nA}, options);
                    torch::Tensor _D = torch::zeros({nC, nU}, options);

                    // Pass raw pointers to FD function
                    mjd_transitionFD(m, d, 1e-8, 1, 
                        _A.data_ptr<double>(), _B.data_ptr<double>(), _C.data_ptr<double>(), _D.data_ptr<double>());

                    // Update matrices (matmul in torch)
                    A = torch::matmul(_A, A);             // [nA, nA]
                    B = torch::matmul(_A, B) + _B;        // [nA, nU]
                    C = torch::matmul(_C, A);             // [nC, nA]
                    D = torch::matmul(_C, B) + _D;        // [nC, nU]
                }

                // Stack blocks just like in Python
                int total_dim = nA + nC;
                auto dydx_block = torch::zeros({total_dim, total_dim}, options);
                dydx_block.index_put_({torch::indexing::Slice(0, nA), torch::indexing::Slice(0, nA)}, A);
                dydx_block.index_put_({torch::indexing::Slice(nA, total_dim), torch::indexing::Slice(0, nA)}, C);

                auto dydu_block = torch::cat({B, D}, 0); // [nA+nC, nU]

                // Save for output or batch stack
                dydx_vec.push_back(dydx_block.clone());
                dydu_vec.push_back(dydu_block.clone());
            }
            m->opt.tolerance    = 1e-10;
            m->opt.ls_tolerance = 1e-10;
            m->opt.disableflags = 0;
        }

        auto states_final      = states.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});         // [B, ...]
        auto sensordatas_final = sensordatas.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});    // [B, ...]

        // Apply qpos_to_axis_angle (your utility function)
        torch::Tensor next_state;
        if (free_joint_flag) {
            next_state = qpos_to_axis_angle(states_final);
        } else {
            next_state = states_final;
        }

        // Concatenate with sensordata along the last dimension
        next_state = torch::cat({next_state, sensordatas_final}, -1);

        // Cast to output dtype/device if needed
        next_state = next_state.to(device, torch::kFloat32);

        // Stack and cast gradients if needed
        torch::Tensor dydx, dydu;
        if (compute_grads) {
            dydx = torch::stack(dydx_vec, 0).to(device, torch::kFloat32); // [B, ..., ...]
            dydu = torch::stack(dydu_vec, 0).to(device, torch::kFloat32); // [B, ..., ...]
            ctx->save_for_backward({dydx, dydu});
        } else {
            dydx = torch::Tensor().to(device, torch::kFloat32); // None in Python
            dydu = torch::Tensor().to(device, torch::kFloat32);
        }

        // Return as tuple
        return {next_state, dydx, dydu};
    }

    static void set_state_and_ctrl(mjModel* m, mjData* d,
        const torch::Tensor& state_row,
        const torch::Tensor& ctrl_row,
        const torch::Tensor& sensordata_row)
    {
        set_state_from_tensor(m, d, state_row);
        set_ctrl_from_tensor(m, d, ctrl_row);
        set_sensordata_from_tensor(m, d, sensordata_row);
    }


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

// Optionally, you can add a custom autograd function here (later step)

// Helper function: forwards arguments to MjStep::apply
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MjStep>(m, "MjStep")
    .def_static("apply", &mjstep_apply,
        py::arg("state"),
        py::arg("ctrl"),
        py::arg("n_steps"),
        py::arg("mj_model_ptr"),
        py::arg("mj_data_ptr"),
        R"pbdoc(
            Differentiable MuJoCo step. Static apply method.
            Usage: next_state, dydx, dydu = MjStep.apply(state, ctrl, n_steps, mj_model_ptr, mj_data_ptr)
        )pbdoc"
    );

}
