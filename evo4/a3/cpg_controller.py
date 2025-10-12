# examples/a3/cpg_controller.py
"""
A tiny CPG controller you can swap in instead of the MLP.

Vector layout (per actuator i):
  A_i, f_i, phi_i, b_i    # amplitude, frequency [Hz], phase [rad], bias
Flat theta length = 4 * O, where O = model.nu

u_i(t) = clip( A_i * sin( 2π f_i * t + phi_i ) + b_i, ±π/2 )

We also expose helpers to (a) infer length, and (b) build a Controller.
"""

from __future__ import annotations
import numpy as np
import mujoco as mj
from ariel.simulation.controllers.controller import Controller


def cpg_param_len(output_size: int) -> int:
    return 4 * int(output_size)


def unpack_cpg_params(theta: np.ndarray, output_size: int):
    theta = np.asarray(theta, dtype=float).ravel()
    need = cpg_param_len(output_size)
    if theta.size < need:
        theta = np.pad(theta, (0, need - theta.size))
    elif theta.size > need:
        theta = theta[:need]
    theta = theta.reshape(output_size, 4)
    A   = theta[:, 0]
    f   = np.abs(theta[:, 1])              # freq must be non-negative
    phi = theta[:, 2]
    b   = np.clip(theta[:, 3], -np.pi/2, np.pi/2)
    # modest clipping on amplitude helps stability
    A   = np.clip(A, 0.0, np.pi/2)
    return A, f, phi, b


def make_cpg_controller_from_theta(theta: np.ndarray, model: mj.MjModel, data: mj.MjData, *, tracker=None) -> Controller:
    """
    Build an ARIEL Controller that produces sinusoidal outputs per actuator.
    """
    O = int(model.nu)
    A, f, phi, b = unpack_cpg_params(theta, O)

    def _cb(m: mj.MjModel, d: mj.MjData, **_kw):
        t = float(d.time)
        # 2π f t + phi
        phase = 2.0 * np.pi * f * t + phi
        u = A * np.sin(phase) + b
        return np.clip(u, -np.pi/2, np.pi/2)

    return Controller(controller_callback_function=_cb, tracker=tracker)
