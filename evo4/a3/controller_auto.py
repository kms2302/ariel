# examples/a3/controller_auto.py
"""
Controller helpers for Assignment 3 (Robot Olympics).
We support:
  1) Random MLP controllers (baseline)
  2) Deterministic MLP controllers decoded from flat genomes (for CMA-ES)
"""

from __future__ import annotations
import numpy as np
import mujoco as mj
from ariel.simulation.controllers.controller import Controller


# -------------------------------
# 1) BASELINE: random MLP per body
# -------------------------------
def init_controller(rng, model: mj.MjModel, data: mj.MjData, hidden: int = 8):
    """
    Simple 2-hidden-layer MLP with random weights per rollout.
    Input  = concat(qpos, qvel)   (depends on body)
    Output = nu (actuators)       (depends on body)
    """
    input_size = data.qpos.size + data.qvel.size
    output_size = int(model.nu)

    w1 = rng.normal(0, 0.5, size=(input_size, hidden))
    w2 = rng.normal(0, 0.5, size=(hidden, hidden))
    w3 = rng.normal(0, 0.5, size=(hidden, output_size))

    def _cb(m: mj.MjModel, d: mj.MjData):
        obs = np.concatenate([d.qpos, d.qvel])
        h1 = np.tanh(obs @ w1)
        h2 = np.tanh(h1 @ w2)
        out = np.tanh(h2 @ w3)
        return np.clip(out, -np.pi/2, np.pi/2)  # safe joint range

    return _cb


# ---------------------------------------------------------
# 2) GENOME-BASED: flat vector -> (W1,W2,W3) -> Controller
# ---------------------------------------------------------
def expected_ctrl_len(input_size: int, hidden: int, output_size: int) -> int:
    """Total #weights of a 2-hidden-layer MLP (I→H, H→H, H→O)."""
    return input_size * hidden + hidden * hidden + hidden * output_size


def unpack_flat_weights(theta: np.ndarray,
                        input_size: int,
                        output_size: int,
                        *,
                        hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map a flat vector into (W1, W2, W3) with shapes:
      W1: (input_size, hidden)
      W2: (hidden, hidden)
      W3: (hidden, output_size)
    If 'theta' is too short/long, pad/truncate to match exactly.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    L1 = input_size * hidden
    L2 = hidden * hidden
    L3 = hidden * output_size
    need = L1 + L2 + L3

    if theta.size < need:
        theta = np.pad(theta, (0, need - theta.size))
    elif theta.size > need:
        theta = theta[:need]

    idx = 0
    W1 = theta[idx:idx+L1].reshape(input_size, hidden); idx += L1
    W2 = theta[idx:idx+L2].reshape(hidden, hidden);     idx += L2
    W3 = theta[idx:idx+L3].reshape(hidden, output_size)
    return W1, W2, W3


def make_mlp_controller_from_weights(W1: np.ndarray,
                                     W2: np.ndarray,
                                     W3: np.ndarray,
                                     *,
                                     tracker=None) -> Controller:
    """
    Wrap fixed MLP weights into an ARIEL Controller (deterministic).
    IMPORTANT: ARIEL's Controller calls tracker.update(...) internally.
               Pass the Tracker you created for this rollout via `tracker=...`.
    """
    in_dim, h = W1.shape
    assert W2.shape == (h, h)
    out_dim = W3.shape[1]

    def _cb(model: mj.MjModel, data: mj.MjData, **_kwargs):
        obs = np.concatenate([data.qpos, data.qvel])

        # Guard against rare dimension mismatches (be defensive)
        if obs.size != in_dim:
            if obs.size > in_dim:
                obs = obs[:in_dim]
            else:
                obs = np.pad(obs, (0, in_dim - obs.size))

        h1 = np.tanh(obs @ W1)
        h2 = np.tanh(h1 @ W2)
        out = np.tanh(h2 @ W3)
        return np.clip(out, -np.pi/2, np.pi/2)

    # Pass the tracker through so Controller.set_control() can call tracker.update(...)
    return Controller(controller_callback_function=_cb, tracker=tracker)
