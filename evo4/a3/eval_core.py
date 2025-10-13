# examples/a3/eval_core.py

from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import mujoco as mj
import numpy as np

from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

from controller_auto import (
    unpack_flat_weights,
    make_mlp_controller_from_weights,
)

try:
    import networkx as nx
except Exception:
    nx = None

SPAWN_POS = [-0.8, 0.0, 0.30]
FINISH_X = 5.42
FINISH_MARGIN = 0.02


def _core_xy(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
    return data.geom_xpos[gid][:2].copy()


def _core_x(model: mj.MjModel, data: mj.MjData) -> float:
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
    return float(data.geom_xpos[gid][0])


def _finite(data: mj.MjData, qacc_cap: float = 2e4) -> bool:
    return (
        np.all(np.isfinite(data.qpos)) and
        np.all(np.isfinite(data.qvel)) and
        np.all(np.isfinite(data.qacc)) and
        float(np.max(np.abs(data.qacc))) < qacc_cap
    )


def _stabilise(model: mj.MjModel) -> None:
    model.opt.timestep   = min(model.opt.timestep, 0.0015)
    model.opt.integrator = mj.mjtIntegrator.mjINT_EULER
    model.opt.solver        = mj.mjtSolver.mjSOL_CG
    model.opt.iterations    = max(model.opt.iterations, 100)
    model.opt.ls_iterations = max(model.opt.ls_iterations, 50)
    model.opt.tolerance     = min(model.opt.tolerance, 1e-8)
    model.dof_damping[:] = np.maximum(model.dof_damping, 2.0) * 2.0
    if getattr(model, "actuator_forcerange", None) is not None:
        lo = model.actuator_forcerange[:, 0]; hi = model.actuator_forcerange[:, 1]
        lo[~np.isfinite(lo)] = -50.0; hi[~np.isfinite(hi)] = 50.0
        model.actuator_forcerange[:, 0] = np.minimum(lo, -5.0)
        model.actuator_forcerange[:, 1] = np.maximum(hi,  5.0)


def _step_for(model: mj.MjModel, data: mj.MjData, seconds: float) -> bool:
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mj.mj_step(model, data)
        if not _finite(data):
            return False
    return True


def _warmup_zero(model: mj.MjModel, data: mj.MjData, seconds: float = 0.60) -> bool:
    old_cb = mj.get_mjcb_control()
    mj.set_mjcb_control(None)
    try:
        return _step_for(model, data, seconds)
    finally:
        mj.set_mjcb_control(old_cb)


class DummyTracker:
    def update(self, *args, **kwargs): 
        return None


def _run_with_theta(model: mj.MjModel, data: mj.MjData, seconds: float, theta: np.ndarray, hidden: int = 8) -> bool:
    """Run rollout with an MLP controller built from flat theta vector."""
    old_cb = mj.get_mjcb_control()
    in_size = data.qpos.size + data.qvel.size
    out_size = int(model.nu)

    W1, W2, W3 = unpack_flat_weights(theta, in_size, out_size, hidden=hidden)
    ctrl = make_mlp_controller_from_weights(W1, W2, W3, tracker=DummyTracker())

    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    try:
        return _step_for(model, data, seconds)
    finally:
        mj.set_mjcb_control(old_cb)


def _ensure_graph(g_like: Any):
    if nx is not None and isinstance(g_like, dict):
        if "edges" in g_like and "links" not in g_like:
            g_like["links"] = g_like["edges"]
        return nx.node_link_graph(g_like, directed=True, multigraph=False)
    return g_like


def _reached_finish(model: mj.MjModel, data: mj.MjData, finish_x: float) -> bool:
    return _core_x(model, data) >= (finish_x - FINISH_MARGIN)


def evaluate_once_graph(
    rng: np.random.Generator,
    *,
    robot_graph: Any,
    duration_s: float = 15.0,
    controller_vec: Path | None = None,
) -> dict[str, Any]:

    world = OlympicArena()
    g = _ensure_graph(robot_graph)
    core = construct_mjspec_from_graph(g)
    world.spawn(core.spec, position=SPAWN_POS)

    model = world.spec.compile()
    data  = mj.MjData(model)
    _stabilise(model)
    mj.mj_resetData(model, data)

    start_xy = _core_xy(model, data)

    nu = int(model.nu)
    if nu <= 0:
        return {"fitness": 0.0, "passed": False, "nu": nu,
                "traj": [start_xy.tolist(), start_xy.tolist()],
                "graph": robot_graph, "reached_finish": False}

    if not _warmup_zero(model, data, 0.60):
        return {"fitness": 0.0, "passed": False, "nu": nu,
                "traj": [start_xy.tolist(), start_xy.tolist()],
                "graph": robot_graph, "reached_finish": False}

    # Load or random theta
    if controller_vec:
        theta = np.load(controller_vec)
    else:
        theta = np.random.normal(0, 0.5, size=(nu * 10))

    _run_with_theta(model, data, duration_s, theta)

    end_xy = _core_xy(model, data)
    dist_xy = float(np.linalg.norm(end_xy - start_xy))

    return {
        "fitness": dist_xy,
        "passed": True,
        "nu": nu,
        "traj": [start_xy.tolist(), end_xy.tolist()],
        "graph": robot_graph,
        "theta": theta.tolist(),
        "reached_finish": _reached_finish(model, data, FINISH_X),
    }

if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, required=True, help="Path to body JSON graph")
    ap.add_argument("--controller", type=str, default=None, help="Optional path to controller .npy file")
    ap.add_argument("--duration", type=float, default=15.0, help="Simulation duration (seconds)")
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(42)

    import json as js
    with open(args.graph, "r") as f:
        graph_json = js.load(f)

    result = evaluate_once_graph(
        rng,
        robot_graph=graph_json,
        duration_s=args.duration,
        controller_vec=args.controller,
    )

    print(json.dumps(result, indent=2))