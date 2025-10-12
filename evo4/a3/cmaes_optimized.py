# examples/a3/cmaes_optimized.py
"""
CMA-ES Alternating Body–Controller Optimization + QoL:
- culling (kill non-learners)
- multi-spawn learning
- dynamic simulation duration
- resume from checkpoints
- optional CPG controller (fast learner) or MLP (default)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from pathlib import Path
import argparse
import re
import json

import numpy as np
import pandas as pd
import mujoco as mj
import cma

try:
    import wandb
except Exception:
    wandb = None  # optional

try:
    import networkx as nx
except Exception:
    nx = None

if TYPE_CHECKING:
    from networkx import DiGraph

# ---------- ARIEL & local imports ----------
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena

from examples.a3.params import POP_SIZE, GENERATIONS, SIGMA_INIT, ENTITY, PROJECT, CONFIG as BASE_CONFIG
from examples.a3.body_evo import (
    random_genotype, decode_to_graph, flatten_genotype, unflatten_genotype,
)
from examples.a3.spawns import cycle_spawns
from examples.a3.culling import cull_one

from examples.a3.controller_auto import (
    unpack_flat_weights, make_mlp_controller_from_weights,
)
from examples.a3.cpg_controller import (
    make_cpg_controller_from_theta, cpg_param_len,
)

# ----------------- Culling (stricter + faster) -----------------
CULL_SECONDS   = 4.0
CULL_THRESHOLD = 0.02   # let “slow starters” survive, kill tiny wiggles

# ----------------- Simulation times (more time to finish) -----------------
SHORT_SECONDS  = 30.0
MID_SECONDS    = 90.0
LONG_SECONDS   = 200.0
THRESH_RUGGED  = -5.0   # once better than this, give MID time
THRESH_FINISH  = -4.5   # once better than this, give LONG time

# ---------- Output folders ----------
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD   = Path.cwd()
DATA  = CWD / "__data__" / SCRIPT_NAME
WBART = CWD / "wandb_artifacts"
DATA.mkdir(parents=True, exist_ok=True)
WBART.mkdir(parents=True, exist_ok=True)

# ---- randomness (seed & RNG) ----
SEED = 42
RNG  = np.random.default_rng(SEED)

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

def _seconds_for(best_f_overall: float) -> float:
    if best_f_overall > THRESH_FINISH:
        return LONG_SECONDS
    if best_f_overall > THRESH_RUGGED:
        return MID_SECONDS
    return SHORT_SECONDS

# parse checkpoints available
def _latest_round(prefix: str) -> int:
    max_r = 0
    for p in DATA.glob(f"{prefix}_round*.npy"):
        m = re.search(r"_round(\d+)\.npy$", p.name)
        if m:
            max_r = max(max_r, int(m.group(1)))
    for p in DATA.glob(f"{prefix}_round*.json"):
        m = re.search(r"_round(\d+)\.json$", p.name)
        if m:
            max_r = max(max_r, int(m.group(1)))
    return max_r

# Mutable holder for duration selected this generation
_CURRENT_DURATION = [SHORT_SECONDS]

# ---------- tiny helpers (avoid circular imports) ----------
def ensure_graph(g):
    """Accept NetworkX graphs, or node-link dicts with 'links' or 'edges'."""
    if isinstance(g, dict):
        if nx is None:
            raise RuntimeError("networkx is required to resume from JSON. pip install networkx")
        obj = dict(g)
        if ("links" not in obj) and ("edges" in obj):
            obj["links"] = obj["edges"]  # normalize for node_link_graph
        return nx.node_link_graph(obj, directed=True, multigraph=False)
    return g

def get_in_out_sizes(robot_graph) -> tuple[int, int]:
    robot_graph = ensure_graph(robot_graph)
    world = OlympicArena()
    core  = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, position=[-0.8, 0.0, 0.30])
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)
    return input_size, output_size

def init_param_vec(rng: np.random.Generator, input_size: int, hidden: int, output_size: int) -> np.ndarray:
    W1 = rng.normal(0, 0.5, size=(input_size, hidden))
    W2 = rng.normal(0, 0.5, size=(hidden, hidden))
    W3 = rng.normal(0, 0.5, size=(hidden, output_size))
    return np.concatenate([W1.ravel(), W2.ravel(), W3.ravel()])

def _load_graph_json(path: Path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and (("nodes" in obj) and ("links" in obj or "edges" in obj)):
        if nx is None:
            raise RuntimeError("networkx is required to resume from JSON. pip install networkx")
        if ("links" not in obj) and ("edges" in obj):
            obj["links"] = obj["edges"]  # normalize
        return nx.node_link_graph(obj, directed=True, multigraph=False)
    return obj

# ---------- Controller builders ----------
def build_mlp_controller(graph, theta: Optional[np.ndarray], hidden: int, tracker, spawn):
    world = OlympicArena()
    core  = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=list(spawn))
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)

    if theta is None:
        W1 = RNG.normal(0, 0.5, size=(input_size, hidden))
        W2 = RNG.normal(0, 0.5, size=(hidden, hidden))
        W3 = RNG.normal(0, 0.5, size=(hidden, output_size))
    else:
        W1, W2, W3 = unpack_flat_weights(theta, input_size, output_size, hidden=hidden)

    ctrl = make_mlp_controller_from_weights(W1, W2, W3, tracker=tracker)
    return model, data, ctrl

def build_cpg_controller(graph, theta: Optional[np.ndarray], tracker, spawn):
    world = OlympicArena()
    core  = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=list(spawn))
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    # wider, walk-friendlier defaults
    if theta is None:
        O = int(model.nu)
        A   = RNG.uniform(0.2, 0.6, size=O)       # bigger swings
        f   = RNG.uniform(0.8, 3.0, size=O)       # allow faster osc
        phi = RNG.uniform(-np.pi, np.pi, size=O)
        b   = RNG.uniform(-0.3, 0.3, size=O)      # allow offset
        theta = np.stack([A, f, phi, b], axis=1).ravel()

    ctrl = make_cpg_controller_from_theta(theta, model, data, tracker=tracker)
    return model, data, ctrl

# ---------- One rollout ----------
def experiment(graph, theta: Optional[np.ndarray], *, hidden: int, ctrl_mode: str, spawn: tuple[float,float,float]) -> float:
    graph = ensure_graph(graph)
    world_tmp = OlympicArena()
    core_tmp  = construct_mjspec_from_graph(graph)
    world_tmp.spawn(core_tmp.spec, position=list(spawn))
    model_tmp = world_tmp.spec.compile()
    data_tmp  = mj.MjData(model_tmp)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world_tmp.spec, data_tmp)

    if ctrl_mode == "cpg":
        model, data, ctrl = build_cpg_controller(graph, theta, tracker, spawn)
    else:
        model, data, ctrl = build_mlp_controller(graph, theta, hidden, tracker, spawn)

    old = mj.get_mjcb_control()
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    try:
        simple_runner(model, data, duration=_CURRENT_DURATION[0])
    finally:
        mj.set_mjcb_control(old)

    return fitness_function(tracker.history["xpos"][0])

# ---------- CMA: evolve body ----------
def find_best_body(controller_vec: Optional[np.ndarray], round_idx: int, x0_vec: Optional[np.ndarray],
                   *, ctrl_mode: str, hidden: int):
    run = None
    if wandb:
        cfg = {**BASE_CONFIG, "ctrl_mode": ctrl_mode, "phase": "body"}
        run = wandb.init(entity=ENTITY, project=PROJECT,
                         name=f"CMA-body-round{round_idx}", config=cfg)

    if x0_vec is None:
        body_vec0 = flatten_genotype(random_genotype(RNG))
        sigma0 = SIGMA_INIT
    else:
        body_vec0 = x0_vec.astype(float, copy=True)
        sigma0 = max(0.2, SIGMA_INIT * 0.6)

    es = cma.CMAEvolutionStrategy(
        body_vec0,
        sigma0,
        {'popsize': POP_SIZE, 'seed': SEED + round_idx, 'verbose': -9}
    )

    best_graph, best_vec = None, None
    best_f_overall = -np.inf
    rows: list[dict] = []
    spawns = cycle_spawns(order=("flat_start","rugged_mid"))

    for g in range(GENERATIONS):
        _CURRENT_DURATION[0] = _seconds_for(best_f_overall)
        pop = es.ask()
        losses, gen_f = [], []

        for v in pop:
            graph = decode_to_graph(unflatten_genotype(v))
            spawn = next(spawns)

            # Cull “statues”
            disp, alive = cull_one(graph, RNG, seconds=CULL_SECONDS, spawn=spawn)
            if (not alive) or (disp < CULL_THRESHOLD):
                losses.append(+1e3)
                gen_f.append(-1e3)
                continue

            f = experiment(graph, controller_vec, hidden=hidden, ctrl_mode=ctrl_mode, spawn=spawn)
            losses.append(-f)
            gen_f.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_vec = v.copy()
                best_graph = graph

        es.tell(pop, losses)
        row = {
            "gen": g,
            "duration": _CURRENT_DURATION[0],
            "best_f_in_gen": float(np.max(gen_f)) if gen_f else -1e3,
            "best_f_overall": float(best_f_overall),
        }
        rows.append(row)
        if run:
            run.log(row, step=g)

    if best_graph is not None:
        save_graph_as_json(best_graph, DATA / f"best_body_round{round_idx}.json")
        np.save(DATA / f"best_body_vec_round{round_idx}.npy", best_vec)

    _save_table(pd.DataFrame(rows), WBART / f"body_round{round_idx}.parquet")
    if run: run.finish()
    return best_graph, best_vec, best_f_overall

# ---------- CMA: evolve controller ----------
def find_best_controller(body_graph, round_idx: int, x0_vec: Optional[np.ndarray],
                         *, ctrl_mode: str, hidden: int):
    body_graph = ensure_graph(body_graph)
    run = None
    if wandb:
        cfg = {**BASE_CONFIG, "ctrl_mode": ctrl_mode, "phase": "controller"}
        run = wandb.init(entity=ENTITY, project=PROJECT,
                         name=f"CMA-ctrl-round{round_idx}", config=cfg)

    in_size, out_size = get_in_out_sizes(body_graph)
    if ctrl_mode == "cpg":
        _ = cpg_param_len(out_size)
        if x0_vec is None:
            A   = RNG.uniform(0.05, 0.3, size=out_size)
            f   = RNG.uniform(0.5, 2.0, size=out_size)
            phi = RNG.uniform(-np.pi, np.pi, size=out_size)
            b   = RNG.uniform(-0.2, 0.2, size=out_size)
            ctrl0 = np.stack([A, f, phi, b], axis=1).ravel()
        else:
            ctrl0 = x0_vec.astype(float, copy=True)
        sigma0 = max(1.0, SIGMA_INIT)  # more exploration for CPG
    else:
        ctrl0 = init_param_vec(RNG, in_size, hidden, out_size)
        if x0_vec is not None:
            ctrl0 = x0_vec.astype(float, copy=True)
        sigma0 = SIGMA_INIT

    es = cma.CMAEvolutionStrategy(
        ctrl0,
        sigma0,
        {'popsize': POP_SIZE, 'seed': SEED + 1000 + round_idx, 'verbose': -9}
    )

    best_vec, best_f_overall = None, -np.inf
    rows: list[dict] = []
    spawns = cycle_spawns(order=("flat_start","rugged_mid"))

    for g in range(GENERATIONS):
        _CURRENT_DURATION[0] = _seconds_for(best_f_overall)
        pop = es.ask()
        losses, gen_f = [], []

        for theta in pop:
            spawn = next(spawns)
            f = experiment(body_graph, theta, hidden=hidden, ctrl_mode=ctrl_mode, spawn=spawn)
            losses.append(-f)
            gen_f.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_vec = theta.copy()

        es.tell(pop, losses)
        row = {
            "gen": g,
            "duration": _CURRENT_DURATION[0],
            "best_f_in_gen": float(np.max(gen_f)) if gen_f else -1e3,
            "best_f_overall": float(best_f_overall),
        }
        rows.append(row)
        if run:
            run.log(row, step=g)

    if best_vec is not None:
        np.save(DATA / f"best_ctrl_vec_round{round_idx}.npy", best_vec)

    _save_table(pd.DataFrame(rows), WBART / f"ctrl_round{round_idx}.parquet")
    if run: run.finish()
    return best_vec, best_f_overall

# ---------- Alternating loop with resume ----------
def alternating_main(rounds: int, ctrl_mode: str, hidden: int, resume: bool):
    start_round = 0
    best_body_vec = None
    best_ctrl_vec = None
    best_body_graph = None

    f_body = float("-inf")
    f_ctrl = float("-inf")

    if resume:
        last_body_r = _latest_round("best_body_vec")
        last_ctrl_r = _latest_round("best_ctrl_vec")
        start_round = max(last_body_r, last_ctrl_r)
        if start_round > 0:
            print(f"[resume] detected latest round = {start_round}")
            body_json = DATA / f"best_body_round{start_round}.json"
            if body_json.exists():
                best_body_graph = _load_graph_json(body_json)
            vec_path = DATA / f"best_body_vec_round{start_round}.npy"
            if vec_path.exists():
                best_body_vec = np.load(vec_path)
            ctrl_path = DATA / f"best_ctrl_vec_round{start_round}.npy"
            if ctrl_path.exists():
                best_ctrl_vec = np.load(ctrl_path)
        # ensure resumed graph is a NetworkX DiGraph
        if best_body_graph is not None:
            best_body_graph = ensure_graph(best_body_graph)
        print(f"[resume] continuing from round {start_round+1}")

    # Ensure we always do at least one more round than the checkpoint we resumed from
    total_rounds = max(rounds, start_round + 1)

    print(f"Alternating optimization for {rounds} rounds (ctrl={ctrl_mode}, hidden={hidden})")

    # Round 0: choose a body using random controllers
    if start_round < 0 or best_body_graph is None:
        best_body_graph, best_body_vec, f_body = find_best_body(
            controller_vec=None, round_idx=0, x0_vec=best_body_vec,
            ctrl_mode=ctrl_mode, hidden=hidden)
    else:
        f_body = -np.inf
        print("[resume] using loaded body for round 0")

    for r in range(start_round + 1, total_rounds + 1):
        print(f"\n=== Round {r} ===")
        best_ctrl_vec, f_ctrl = find_best_controller(
            best_body_graph, round_idx=r, x0_vec=best_ctrl_vec,
            ctrl_mode=ctrl_mode, hidden=hidden)
        print(f"[Round {r}] controller best f = {f_ctrl:.3f}")

        best_body_graph, best_body_vec, f_body = find_best_body(
            controller_vec=best_ctrl_vec, round_idx=r, x0_vec=best_body_vec,
            ctrl_mode=ctrl_mode, hidden=hidden)
        print(f"[Round {r}] body best f = {f_body:.3f}")

    print("\nDONE.")
    print(f"Final body f = {f_body:.3f} | Final ctrl f = {f_ctrl:.3f}")
    save_graph_as_json(best_body_graph, DATA / "final_best_body.json")
    if best_ctrl_vec is not None:
        np.save(DATA / "final_best_ctrl.npy", best_ctrl_vec)

# ---------- CLI ----------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="number of alternating rounds")
    ap.add_argument("--controller", choices=["mlp","cpg"], default="mlp", help="controller type")
    ap.add_argument("--hidden", type=int, default=8, help="MLP hidden size (ignored for CPG)")
    ap.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    alternating_main(rounds=args.rounds, ctrl_mode=args.controller, hidden=args.hidden, resume=args.resume)
