#ifndef MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_PARALLEL_H_
#define MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_PARALLEL_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif


MJAPI void mjd_stepFD_parallel(
    const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
    mjtNum* DyDq, mjtNum* DyDv, mjtNum* DyDa, mjtNum* DyDu,
    mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa, mjtNum* DsDu,
    mjData* d_ctrl, mjData* d_act, mjData* d_vel, mjData* d_pos);

MJAPI void mjd_transitionFD_parallel(const mjModel* m, mjData* d, mjtNum eps, mjtByte centered,
                            mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_FD_H_
