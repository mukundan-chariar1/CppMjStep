// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine/engine_derivative_fd.h"

#include <stddef.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include "engine/engine_forward.h"
#include "engine/engine_io.h"
#include "engine/engine_inverse.h"
#include "engine/engine_macro.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

//--------------------------- finite-differencing utility functions --------------------------------

// get state=[qpos; qvel; act] and optionally sensordata
static void getState(const mjModel* m, const mjData* d, mjtNum* state, mjtNum* sensordata) {
  mj_getState(m, d, state, mjSTATE_PHYSICS);
  if (sensordata) {
    mju_copy(sensordata, d->sensordata, m->nsensordata);
  }
}



// dx = (x2 - x1) / h
static void diff(mjtNum* restrict dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n) {
  mjtNum inv_h = 1/h;
  for (int i=0; i < n; i++) {
    dx[i] = inv_h * (x2[i] - x1[i]);
  }
}



// finite-difference two state vectors ds = (s2 - s1) / h
static void stateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2, mjtNum h) {
  int nq = m->nq, nv = m->nv, na = m->na;

  if (nq == nv) {
    diff(ds, s1, s2, h, nq+nv+na);
  } else {
    mj_differentiatePos(m, ds, h, s1, s2);
    diff(ds+nv, s1+nq, s2+nq, h, nv+na);
  }
}



// finite-difference two vectors, forward, backward or centered
static void clampedDiff(mjtNum* dx, const mjtNum* x, const mjtNum* x_plus, const mjtNum* x_minus,
                        mjtNum h, int nx) {
  if (x_plus && !x_minus) {
    // forward differencing
    diff(dx, x, x_plus, h, nx);
  } else if (!x_plus && x_minus) {
    // backward differencing
    diff(dx, x_minus, x, h, nx);
  } else if (x_plus && x_minus) {
    // centered differencing
    diff(dx, x_plus, x_minus, 2*h, nx);
  } else {
    // differencing failed, write zeros
    mju_zero(dx, nx);
  }
}



// finite-difference two state vectors, forward, backward or centered
static void clampedStateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s, const mjtNum* s_plus,
                             const mjtNum* s_minus, mjtNum h) {
  if (s_plus && !s_minus) {
    // forward differencing
    stateDiff(m, ds, s, s_plus, h);
  } else if (!s_plus && s_minus) {
    // backward differencing
    stateDiff(m, ds, s_minus, s, h);
  } else if (s_plus && s_minus) {
    // centered differencing
    stateDiff(m, ds, s_minus, s_plus, 2*h);
  } else {
    // differencing failed, write zeros
    mju_zero(ds, 2*m->nv + m->na);
  }
}



// check if two numbers are inside a given range
static int inRange(const mjtNum x1, const mjtNum x2, const mjtNum* range) {
  return x1 >= range[0] && x1 <= range[1] &&
         x2 >= range[0] && x2 <= range[1];
}



// advance simulation using control callback, skipstage is mjtStage
void mj_stepSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor) {
  TM_START;

  // common to all integrators
  mj_checkPos(m, d);
  mj_checkVel(m, d);
  mj_forwardSkip(m, d, skipstage, skipsensor);
  mj_checkAcc(m, d);

  // compare forward and inverse solutions if enabled
  if (mjENABLED(mjENBL_FWDINV)) {
    mj_compareFwdInv(m, d);
  }

  // use selected integrator
  switch ((mjtIntegrator) m->opt.integrator) {
  case mjINT_EULER:
    mj_EulerSkip(m, d, skipstage >= mjSTAGE_POS);
    break;

  case mjINT_RK4:
    // ignore skipstage
    mj_RungeKutta(m, d, 4);
    break;

  case mjINT_IMPLICIT:
  case mjINT_IMPLICITFAST:
    mj_implicitSkip(m, d, skipstage >= mjSTAGE_VEL);
    break;

  default:
    mjERROR("invalid integrator");
  }

  TM_END(mjTIMER_STEP);
}

static inline void nudgeControls(
    const mjModel* m, mjData* d_ctrl,
    mjtNum eps, mjtByte flg_centered,
    const mjtNum* fullstate, const unsigned int restore_spec,
    const mjtNum* ctrl, const mjtNum* next, const mjtNum* sensor,   // baseline NEXT
    const int skipsensor,                                        // 0 if you need sensors
    mjtNum* DyDu, mjtNum* DsDu)                            // outputs: (nu x ndx), (nu x ns)
{
  const int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
  const int ndx = 2*nv + na;

  mj_markStack(d_ctrl);

  mjtNum *next_plus  = mjSTACKALLOC(d_ctrl, nq+nv+na, mjtNum);  // forward-nudged next state
  mjtNum *next_minus = mjSTACKALLOC(d_ctrl, nq+nv+na, mjtNum);  // backward-nudged next state
  mjtNum *sensor_plus  = skipsensor ? NULL : mjSTACKALLOC(d_ctrl, ns, mjtNum);  // forward-nudged
  mjtNum *sensor_minus = skipsensor ? NULL : mjSTACKALLOC(d_ctrl, ns, mjtNum);  // backward-nudged

  for (int i=0; i < nu; i++) {
      int limited = m->actuator_ctrllimited[i];
      // nudge forward, if possible given ctrlrange
      int nudge_fwd = !limited || inRange(ctrl[i], ctrl[i]+eps, m->actuator_ctrlrange+2*i);
      if (nudge_fwd) {
        // nudge forward
        d_ctrl->ctrl[i] += eps;

        // step, get nudged output
        mj_stepSkip(m, d_ctrl, mjSTAGE_VEL, skipsensor);
        getState(m, d_ctrl, next_plus, sensor_plus);

        // reset
        mj_setState(m, d_ctrl, fullstate, restore_spec);
      }

      // nudge backward, if possible given ctrlrange
      int nudge_back = (flg_centered || !nudge_fwd) &&
                       (!limited || inRange(ctrl[i]-eps, ctrl[i], m->actuator_ctrlrange+2*i));
      if (nudge_back) {
        // nudge backward
        d_ctrl->ctrl[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d_ctrl, mjSTAGE_VEL, skipsensor);
        getState(m, d_ctrl, next_minus, sensor_minus);

        // reset
        mj_setState(m, d_ctrl, fullstate, restore_spec);
      }

      // difference states
      if (DyDu) {
        clampedStateDiff(m, DyDu+i*ndx, next, nudge_fwd ? next_plus : NULL,
                         nudge_back ? next_minus : NULL, eps);
      }

      // difference sensors
      if (DsDu) {
        clampedDiff(DsDu+i*ns, sensor, nudge_fwd ? sensor_plus : NULL,
                    nudge_back ? sensor_minus : NULL, eps, ns);
      }
    }

    mj_freeStack(d_ctrl);
}

static inline void nudgeActuation(const mjModel* m, mjData* d_act,
    mjtNum eps, mjtByte flg_centered,
    const mjtNum* fullstate, const unsigned int restore_spec,
    const mjtNum* next, const mjtNum* sensor,   // baseline NEXT
    const int skipsensor,                                        // 0 if you need sensors
    mjtNum* DyDa, mjtNum* DsDa)
{
    const int nq = m->nq, nv = m->nv, na = m->na, ns = m->nsensordata;
    const int ndx = 2*nv + na;

    mj_markStack(d_act);

    mjtNum *next_plus  = mjSTACKALLOC(d_act, nq+nv+na, mjtNum);  // forward-nudged next state
    mjtNum *next_minus = mjSTACKALLOC(d_act, nq+nv+na, mjtNum);  // backward-nudged next state
    mjtNum *sensor_plus  = skipsensor ? NULL : mjSTACKALLOC(d_act, ns, mjtNum);  // forward-nudged
    mjtNum *sensor_minus = skipsensor ? NULL : mjSTACKALLOC(d_act, ns, mjtNum);  // backward-nudged

    for (int i=0; i < na; i++) {
      // nudge forward
      d_act->act[i] += eps;

      // step, get nudged output
      mj_stepSkip(m, d_act, mjSTAGE_VEL, skipsensor);
      getState(m, d_act, next_plus, sensor_plus);

      // reset
      mj_setState(m, d_act, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge backward
        d_act->act[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d_act, mjSTAGE_VEL, skipsensor);
        getState(m, d_act, next_minus, sensor_minus);

        // reset
        mj_setState(m, d_act, fullstate, restore_spec);
      }

      // difference states
      if (DyDa) {
        if (!flg_centered) {
          stateDiff(m, DyDa+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDa+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDa) {
        if (!flg_centered) {
          diff(DsDa+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDa+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }

    mj_freeStack(d_act);
}

static inline void nudgeVelocities(const mjModel* m, mjData* d_vel,
    mjtNum eps, mjtByte flg_centered,
    const mjtNum* fullstate, const unsigned int restore_spec,
    const mjtNum* next, const mjtNum* sensor,   // baseline NEXT
    const int skipsensor,                                        // 0 if you need sensors
    mjtNum* DyDv, mjtNum* DsDv)
{
    const int nq = m->nq, nv = m->nv, na = m->na, ns = m->nsensordata;
    const int ndx = 2*nv + na;

    mj_markStack(d_vel);

    mjtNum *next_plus  = mjSTACKALLOC(d_vel, nq+nv+na, mjtNum);  // forward-nudged next state
    mjtNum *next_minus = mjSTACKALLOC(d_vel, nq+nv+na, mjtNum);  // backward-nudged next state
    mjtNum *sensor_plus  = skipsensor ? NULL : mjSTACKALLOC(d_vel, ns, mjtNum);  // forward-nudged
    mjtNum *sensor_minus = skipsensor ? NULL : mjSTACKALLOC(d_vel, ns, mjtNum);  // backward-nudged

    for (int i=0; i < nv; i++) {
      // nudge forward
      d_vel->qvel[i] += eps;

      // step, get nudged output
      mj_stepSkip(m, d_vel, mjSTAGE_POS, skipsensor);
      getState(m, d_vel, next_plus, sensor_plus);

      // reset
      mj_setState(m, d_vel, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge
        d_vel->qvel[i] -= eps;

        // step, get nudged output
        mj_stepSkip(m, d_vel, mjSTAGE_POS, skipsensor);
        getState(m, d_vel, next_minus, sensor_minus);

        // reset
        mj_setState(m, d_vel, fullstate, restore_spec);
      }

      // difference states
      if (DyDv) {
        if (!flg_centered) {
          stateDiff(m, DyDv+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDv+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDv) {
        if (!flg_centered) {
          diff(DsDv+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDv+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }

    mj_freeStack(d_vel);
}

static inline void nudgePositions(const mjModel* m, mjData* d_pos,
    mjtNum eps, mjtByte flg_centered,
    const mjtNum* fullstate, const unsigned int restore_spec,
    const mjtNum* next, const mjtNum* sensor,   // baseline NEXT
    const int skipsensor,                                        // 0 if you need sensors
    mjtNum* DyDq, mjtNum* DsDq)
{
    const int nq = m->nq, nv = m->nv, na = m->na, ns = m->nsensordata;
    const int ndx = 2*nv + na;

    mj_markStack(d_pos);

    mjtNum *next_plus  = mjSTACKALLOC(d_pos, nq+nv+na, mjtNum);  // forward-nudged next state
    mjtNum *next_minus = mjSTACKALLOC(d_pos, nq+nv+na, mjtNum);  // backward-nudged next state
    mjtNum *sensor_plus  = skipsensor ? NULL : mjSTACKALLOC(d_pos, ns, mjtNum);  // forward-nudged
    mjtNum *sensor_minus = skipsensor ? NULL : mjSTACKALLOC(d_pos, ns, mjtNum);  // backward-nudged

    mjtNum *dpos  = mjSTACKALLOC(d_pos, nv, mjtNum);  // allocate position perturbation

    for (int i=0; i < nv; i++) {
      // nudge forward
      mju_zero(dpos, nv);
      dpos[i] = 1;
      mj_integratePos(m, d_pos->qpos, dpos, eps);

      // step, get nudged output
      mj_stepSkip(m, d_pos, mjSTAGE_NONE, skipsensor);
      getState(m, d_pos, next_plus, sensor_plus);

      // reset
      mj_setState(m, d_pos, fullstate, restore_spec);

      // nudge backward
      if (flg_centered) {
        // nudge backward
        mju_zero(dpos, nv);
        dpos[i] = 1;
        mj_integratePos(m, d_pos->qpos, dpos, -eps);

        // step, get nudged output
        mj_stepSkip(m, d_pos, mjSTAGE_NONE, skipsensor);
        getState(m, d_pos, next_minus, sensor_minus);

        // reset
        mj_setState(m, d_pos, fullstate, restore_spec);
      }

      // difference states
      if (DyDq) {
        if (!flg_centered) {
          stateDiff(m, DyDq+i*ndx, next, next_plus, eps);
        } else {
          stateDiff(m, DyDq+i*ndx, next_minus, next_plus, 2*eps);
        }
      }

      // difference sensors
      if (DsDq) {
        if (!flg_centered) {
          diff(DsDq+i*ns, sensor, sensor_plus, eps, ns);
        } else {
          diff(DsDq+i*ns, sensor_minus, sensor_plus, 2*eps, ns);
        }
      }
    }

    mj_freeStack(d_pos);
}

// finite differenced Jacobian of  (next_state, sensors) = mj_step(state, control)
//   all outputs are optional
//   output dimensions (transposed w.r.t Control Theory convention):
//     DyDq: (nv x 2*nv+na)
//     DyDv: (nv x 2*nv+na)
//     DyDa: (na x 2*nv+na)
//     DyDu: (nu x 2*nv+na)
//     DsDq: (nv x nsensordata)
//     DsDv: (nv x nsensordata)
//     DsDa: (na x nsensordata)
//     DsDu: (nu x nsensordata)
//   single-letter shortcuts:
//     inputs: q=qpos, v=qvel, a=act, u=ctrl
//     outputs: y=next_state (concatenated next qpos, qvel, act), s=sensordata

void mjd_stepFD_parallel(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                mjtNum* DyDq, mjtNum* DyDv, mjtNum* DyDa, mjtNum* DyDu,
                mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa, mjtNum* DsDu,
                mjData* d_ctrl, mjData* d_act, mjData* d_vel, mjData* d_pos) {
  int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
  
  mj_markStack(d);

  // state to restore after finite differencing
  unsigned int restore_spec = mjSTATE_FULLPHYSICS | mjSTATE_CTRL;
  restore_spec |= mjDISABLED(mjDSBL_WARMSTART) ? 0 : mjSTATE_WARMSTART;

  mjtNum *fullstate  = mjSTACKALLOC(d, mj_stateSize(m, restore_spec), mjtNum);
  mjtNum *state      = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // current state
  mjtNum *next       = mjSTACKALLOC(d, nq+nv+na, mjtNum);  // next state

  // sensors
  int skipsensor = !DsDq && !DsDv && !DsDa && !DsDu;
  mjtNum *sensor       = skipsensor ? NULL : mjSTACKALLOC(d, ns, mjtNum);  // sensor values

  // controls
  mjtNum *ctrl = mjSTACKALLOC(d, nu, mjtNum);

  // save current inputs
  mj_getState(m, d, fullstate, restore_spec);
  mju_copy(ctrl, d->ctrl, nu);
  getState(m, d, state, NULL);

  // step input
  mj_stepSkip(m, d, mjSTAGE_NONE, skipsensor);

  // save output
  getState(m, d, next, sensor);

  // restore input
  mj_setState(m, d, fullstate, restore_spec);

  if ((DyDu || DsDu) && d_ctrl) {
    mj_setState(m, d_ctrl, fullstate, restore_spec);
    if (!skipsensor) mju_copy(d_ctrl->sensordata, sensor, ns);
  }
  if ((DyDa || DsDa) && d_act) {
    mj_setState(m, d_act, fullstate, restore_spec);
    if (!skipsensor) mju_copy(d_act->sensordata, sensor, ns);
  }
  if ((DyDv || DsDv) && d_vel) {
    mj_setState(m, d_vel, fullstate, restore_spec);
    if (!skipsensor) mju_copy(d_vel->sensordata, sensor, ns);
  }
  if ((DyDq || DsDq) && d_pos) {
    mj_setState(m, d_pos, fullstate, restore_spec);
    if (!skipsensor) mju_copy(d_pos->sensordata, sensor, ns);
  }

  // // finite-difference controls: skip=mjSTAGE_VEL, handle ctrl at range limits
  // if (DyDu || DsDu) {
  //   nudgeControls(
  //     m, d_ctrl,
  //     eps, flg_centered,
  //     /*fullstate*/   fullstate,
  //     /*restore_spec*/restore_spec,
  //     /*ctrl*/        ctrl,   // baseline controls pointer
  //     /*next*/        next,           // baseline NEXT (already computed earlier)
  //     /*sensor*/      sensor,         // baseline sensors (nullptr if not needed)
  //     /*skipsensor*/  skipsensor,
  //     /*DyDu*/        DyDu,           // (nu x ndx)
  //     /*DsDu*/        DsDu);          // (nu x ns) or nullptr
  // }

  // // finite-difference activations: skip=mjSTAGE_VEL
  // if (DyDa || DsDa) {
  //   nudgeActuation(
  //     m, d_act,
  //     eps, flg_centered,
  //     /*fullstate*/   fullstate,
  //     /*restore_spec*/restore_spec,
  //     /*next*/        next,           // baseline NEXT (already computed earlier)
  //     /*sensor*/      sensor,         // baseline sensors (nullptr if not needed)
  //     /*skipsensor*/  skipsensor,
  //     /*DyDu*/        DyDa,           // (nu x ndx)
  //     /*DsDu*/        DsDa);          // (nu x ns) or nullptr
  // }


  // // finite-difference velocities: skip=mjSTAGE_POS
  // if (DyDv || DsDv) {
  //   nudgeVelocities(
  //     m, d_vel,
  //     eps, flg_centered,
  //     /*fullstate*/   fullstate,
  //     /*restore_spec*/restore_spec,
  //     /*next*/        next,           // baseline NEXT (already computed earlier)
  //     /*sensor*/      sensor,         // baseline sensors (nullptr if not needed)
  //     /*skipsensor*/  skipsensor,
  //     /*DyDu*/        DyDv,           // (nu x ndx)
  //     /*DsDu*/        DsDv);          // (nu x ns) or nullptr
  // }

  // // finite-difference positions: skip=mjSTAGE_NONE
  // if (DyDq || DsDq) {
  //   nudgePositions(
  //     m, d_pos,
  //     eps, flg_centered,
  //     /*fullstate*/   fullstate,
  //     /*restore_spec*/restore_spec,
  //     /*next*/        next,           // baseline NEXT (already computed earlier)
  //     /*sensor*/      sensor,         // baseline sensors (nullptr if not needed)
  //     /*skipsensor*/  skipsensor,
  //     /*DyDu*/        DyDq,           // (nu x ndx)
  //     /*DsDu*/        DsDq);          // (nu x ns) or nullptr
  // }

  #pragma omp parallel sections if( (DyDu||DsDu) + (DyDa||DsDa) + (DyDv||DsDv) + (DyDq||DsDq) > 1 ) \
                               default(none) shared(m,eps,flg_centered,restore_spec,ctrl,next,sensor,skipsensor,ns) \
                               shared(DyDu,DsDu,DyDa,DsDa,DyDv,DsDv,DyDq,DsDq) \
                               shared(d_ctrl,d_act,d_vel,d_pos,fullstate)
  {
    #pragma omp section
    {
      if ((DyDu || DsDu) && d_ctrl) {
        nudgeControls(m, d_ctrl, eps, flg_centered,
                      fullstate, restore_spec,
                      ctrl, next, sensor, skipsensor,
                      DyDu, DsDu);
      }
    }
    #pragma omp section
    {
      if ((DyDa || DsDa) && d_act) {
        nudgeActuation(m, d_act, eps, flg_centered,
                       fullstate, restore_spec,
                       next, sensor, skipsensor,
                       DyDa, DsDa);
      }
    }
    #pragma omp section
    {
      if ((DyDv || DsDv) && d_vel) {
        nudgeVelocities(m, d_vel, eps, flg_centered,
                        fullstate, restore_spec,
                        next, sensor, skipsensor,
                        DyDv, DsDv);
      }
    }
    #pragma omp section
    {
      if ((DyDq || DsDq) && d_pos) {
        nudgePositions(m, d_pos, eps, flg_centered,
                       fullstate, restore_spec,
                       next, sensor, skipsensor,
                       DyDq, DsDq);
      }
    }
  } // end parallel sections

  mj_freeStack(d);
}

// finite differenced transition matrices (control theory notation)
//   d(x_next) = A*dx + B*du
//   d(sensor) = C*dx + D*du
//   required output matrix dimensions:
//      A: (2*nv+na x 2*nv+na)
//      B: (2*nv+na x nu)
//      C: (nsensordata x 2*nv+na)
//      D: (nsensordata x nu)
void mjd_transitionFD_parallel(const mjModel* m, mjData* d, mjtNum eps, mjtByte flg_centered,
                      mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D) {
  if (m->opt.integrator == mjINT_RK4) {
    mjERROR("RK4 integrator is not supported");
  }

  int nv = m->nv, na = m->na, nu = m->nu, ns = m->nsensordata;
  int ndx = 2*nv+na;  // row length of state Jacobians

  // stepFD() offset pointers, initialised to NULL
  mjtNum *DyDq, *DyDv, *DyDa, *DsDq, *DsDv, *DsDa;
  DyDq = DyDv = DyDa = DsDq = DsDv = DsDa = NULL;

  mjData* d_ctrl = mj_makeData(m);
  mj_copyData(d_ctrl, m, d);

  mjData* d_act = mj_makeData(m);
  mj_copyData(d_act, m, d);

  mjData* d_vel = mj_makeData(m);
  mj_copyData(d_vel, m, d);

  mjData* d_pos = mj_makeData(m);
  mj_copyData(d_pos, m, d);

  mj_markStack(d);

  // allocate transposed matrices
  mjtNum *AT = A ? mjSTACKALLOC(d, ndx*ndx, mjtNum) : NULL;  // state-transition     (transposed)
  mjtNum *BT = B ? mjSTACKALLOC(d, nu*ndx, mjtNum) : NULL;   // control-transition   (transposed)
  mjtNum *CT = C ? mjSTACKALLOC(d, ndx*ns, mjtNum) : NULL;   // state-observation    (transposed)
  mjtNum *DT = D ? mjSTACKALLOC(d, nu*ns, mjtNum) : NULL;    // control-observation  (transposed)

  // set offset pointers
  if (A) {
    DyDq = AT;
    DyDv = AT+ndx*nv;
    DyDa = AT+ndx*2*nv;
  }

  if (C) {
    DsDq = CT;
    DsDv = CT + ns*nv;
    DsDa = CT + ns*2*nv;
  }

  // get Jacobians
  mjd_stepFD_parallel(m, d, eps, flg_centered, DyDq, DyDv, DyDa, BT, DsDq, DsDv, DsDa, DT, 
                      d_ctrl, d_act, d_vel, d_pos);

  // transpose
  if (A) mju_transpose(A, AT, ndx, ndx);
  if (B) mju_transpose(B, BT, nu, ndx);
  if (C) mju_transpose(C, CT, ndx, ns);
  if (D) mju_transpose(D, DT, nu, ns);

  mj_freeStack(d);

  mj_deleteData(d_ctrl);
  mj_deleteData(d_act);
  mj_deleteData(d_vel);
  mj_deleteData(d_pos);
}