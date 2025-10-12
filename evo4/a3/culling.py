# examples/a3/culling.py
from __future__ import annotations

import numpy as np
import mujoco as mj

from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner

from examples.a3.controller_auto import init_controller

# Default spawn used if none is provided
DEFAULT_SPAWN = (-0.8, 0.0, 0.28)

def _xy_disp(tracker: Tracker) -> float:
    path = np.array(tracker.history["xpos"][0])
    a, b = path[0, :2], path[-1, :2]
    return float(np.linalg.norm(b - a))

def cull_one(graph, rng, *, seconds: float = 2.0, spawn: tuple[float,float,float] | None = None) -> tuple[float, bool]:
    """
    Short rollout with a random MLP controller.
    Returns (xy_displacement_in_m, alive_flag).
    """
    mj.set_mjcb_control(None)
    world = OlympicArena()

    core = construct_mjspec_from_graph(graph)
    pos = list(spawn or DEFAULT_SPAWN)
    world.spawn(core.spec, position=pos)

    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    ctrl_fn = init_controller(rng, model, data)
    from ariel.simulation.controllers.controller import Controller
    ctrl = Controller(controller_callback_function=ctrl_fn, tracker=tracker)

    old = mj.get_mjcb_control()
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    try:
        simple_runner(model, data, duration=seconds)
    finally:
        mj.set_mjcb_control(old)

    disp = _xy_disp(tracker)
    alive = np.isfinite(disp) and disp >= 0.0
    return disp, bool(alive)
