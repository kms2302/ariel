# examples/a3/cmaes_optimized.py
"""
CMA-ES Alternating Body–Controller Optimization
Each round:
  1) evolve controller for the current best body
  2) evolve body using the new best controller
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco as mj
import cma

try:
    import wandb
except Exception:
    wandb = None  # optional; script runs without Weights & Biases

if TYPE_CHECKING:
    from networkx import DiGraph

# ---------- ARIEL & local imports ----------
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena

from examples.a3.body_evo import (
    random_genotype, decode_to_graph, flatten_genotype, unflatten_genotype,
)

from examples.a3.controller_auto import (
    expected_ctrl_len, unpack_flat_weights, make_mlp_controller_from_weights,
)

# (We keep these imports available for the fixed-length genome approach, even if
# this alternating script doesn’t use them directly right now.)
from examples.a3.genome_slicing import (
    slice_body_brain, get_io_sizes_for_graph as _unused_get_io_sizes_for_graph,
    pick_brain_slice_for_this_body, HIDDEN_SIZE,
)

# ---------- Hyperparameters ----------
POP_SIZE     = 16
GENERATIONS  = 16
SIGMA_INIT   = 0.7
SECONDS      = 15.0
SEED         = 42
RNG          = np.random.default_rng(SEED)

# ---------- Output folders ----------
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD   = Path.cwd()
DATA  = CWD / "__data__" / SCRIPT_NAME
WBART = CWD / "wandb_artifacts"
DATA.mkdir(parents=True, exist_ok=True)
WBART.mkdir(parents=True, exist_ok=True)

ENTITY  = "evo4"
PROJECT = "assignment3"
CONFIG  = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
    "Hidden Size": HIDDEN_SIZE,
    "Seconds": SECONDS,
}

# --- lightweight logging helper: parquet if available, else CSV ---
def _save_table(df: pd.DataFrame, out_path: Path) -> str:
    try:
        df.to_parquet(out_path, index=False)
        return str(out_path)
    except Exception:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)

# ---------- Fitness ----------
TARGET_POSITION = np.array([5.0, 0.0, 0.5])

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    p = np.array(history[-1], dtype=float)
    return -float(np.linalg.norm(TARGET_POSITION - p))

# ---------- LOCAL HELPERS (avoid importing from utils to prevent circulars) ----------
def _get_in_out_sizes(body_graph) -> tuple[int, int]:
    """Compile this body and return (input_size, output_size)."""
    world = OlympicArena()
    core  = construct_mjspec_from_graph(body_graph)
    world.spawn(core.spec, position=[-0.8, 0.0, 0.28])
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)
    return input_size, output_size

def _init_param_vec(rng: np.random.Generator, input_size: int, hidden: int, output_size: int) -> np.ndarray:
    """Flat vector of random MLP weights of the right length."""
    L = expected_ctrl_len(input_size, hidden, output_size)
    return rng.normal(0.0, 0.5, size=L)

# ---------- One rollout ----------
def experiment(graph, theta: np.ndarray | None, *, hidden: int = HIDDEN_SIZE) -> float:
    """
    Build model for 'graph', make controller from 'theta' (flat), run SECONDS, return fitness.
    If theta is None, use a random controller (baseline).
    """
    world = OlympicArena()
    core  = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=[-0.8, 0.0, 0.28])
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    # tracker for 'core' geom (Controller will call tracker.update(...) every step)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # sizes for THIS body
    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)

    # pick/create weights
    if theta is None:
        # baseline: random MLP
        W1 = RNG.normal(0, 0.5, size=(input_size, hidden))
        W2 = RNG.normal(0, 0.5, size=(hidden, hidden))
        W3 = RNG.normal(0, 0.5, size=(hidden, output_size))
    else:
        # CMA-ES candidate: flat -> (W1,W2,W3) shaped for THIS body
        W1, W2, W3 = unpack_flat_weights(theta, input_size, output_size, hidden=hidden)

    # IMPORTANT: pass the tracker into the controller so set_control() can call tracker.update(...)
    ctrl = make_mlp_controller_from_weights(W1, W2, W3, tracker=tracker)

    # Install control callback safely, then run
    old = mj.get_mjcb_control()
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    try:
        simple_runner(model, data, duration=SECONDS)
    finally:
        mj.set_mjcb_control(old)

    return fitness_function(tracker.history["xpos"][0])

# ---------- CMA: Evolve body (optionally with a fixed controller) ----------
def find_best_body(controller_vec: np.ndarray | None, round_idx: int):
    run = None
    if wandb:
        run = wandb.init(entity=ENTITY, project=PROJECT,
                         name=f"CMA-body-round{round_idx}", config=CONFIG)

    # start from a random genotype vector
    body_vec0 = flatten_genotype(random_genotype(RNG))
    es = cma.CMAEvolutionStrategy(body_vec0, SIGMA_INIT,
                                  {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9})

    best_graph, best_vec = None, None
    best_f_overall = -np.inf
    rows: list[dict] = []

    for g in range(GENERATIONS):
        pop = es.ask()
        losses, gen_f = [], []

        for v in pop:
            graph = decode_to_graph(unflatten_genotype(v))
            f = experiment(graph, theta=controller_vec)
            losses.append(-f)
            gen_f.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_vec = v.copy()
                best_graph = graph

        es.tell(pop, losses)
        row = {"gen": g, "best_f_in_gen": float(np.max(gen_f)), "best_f_overall": float(best_f_overall)}
        rows.append(row)
        if run:
            run.log({"phase": "body", **row}, step=g)

    if best_graph is not None:
        save_graph_as_json(best_graph, DATA / f"best_body_round{round_idx}.json")
        np.save(DATA / f"best_body_vec_round{round_idx}.npy", best_vec)

    _save_table(pd.DataFrame(rows), WBART / f"body_round{round_idx}.parquet")
    if run: run.finish()
    return best_graph, best_vec, best_f_overall

# ---------- CMA: Evolve controller for a fixed body ----------
def find_best_controller(body_graph, round_idx: int):
    run = None
    if wandb:
        run = wandb.init(entity=ENTITY, project=PROJECT,
                         name=f"CMA-ctrl-round{round_idx}", config=CONFIG)

    # sizes to seed CMA
    in_size, out_size = _get_in_out_sizes(body_graph)
    ctrl0 = _init_param_vec(RNG, in_size, HIDDEN_SIZE, out_size)
    es = cma.CMAEvolutionStrategy(ctrl0, SIGMA_INIT,
                                  {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9})

    best_vec, best_f_overall = None, -np.inf
    rows: list[dict] = []

    for g in range(GENERATIONS):
        pop = es.ask()
        losses, gen_f = [], []

        for theta in pop:
            f = experiment(body_graph, theta=theta)
            losses.append(-f)
            gen_f.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_vec = theta.copy()

        es.tell(pop, losses)
        row = {"gen": g, "best_f_in_gen": float(np.max(gen_f)), "best_f_overall": float(best_f_overall)}
        rows.append(row)
        if run:
            run.log({"phase": "controller", **row}, step=g)

    if best_vec is not None:
        np.save(DATA / f"best_ctrl_vec_round{round_idx}.npy", best_vec)

    _save_table(pd.DataFrame(rows), WBART / f"ctrl_round{round_idx}.parquet")
    if run: run.finish()
    return best_vec, best_f_overall

# ---------- Alternating loop ----------
def alternating_main(rounds: int = 3):
    print(f"Alternating optimization for {rounds} rounds")

    # Round 0: pick a body using random controllers
    best_body_graph, best_body_vec, f_body = find_best_body(controller_vec=None, round_idx=0)
    best_ctrl_vec, f_ctrl = None, -np.inf

    for r in range(1, rounds + 1):
        print(f"\n=== Round {r} ===")
        best_ctrl_vec, f_ctrl = find_best_controller(best_body_graph, round_idx=r)
        print(f"[Round {r}] controller best f = {f_ctrl:.3f}")

        best_body_graph, best_body_vec, f_body = find_best_body(controller_vec=best_ctrl_vec, round_idx=r)
        print(f"[Round {r}] body best f = {f_body:.3f}")

    print("\nDONE.")
    print(f"Final body f = {f_body:.3f} | Final ctrl f = {f_ctrl:.3f}")
    save_graph_as_json(best_body_graph, DATA / "final_best_body.json")
    if best_ctrl_vec is not None:
        np.save(DATA / "final_best_ctrl.npy", best_ctrl_vec)

if __name__ == "__main__":
    alternating_main(rounds=4)
