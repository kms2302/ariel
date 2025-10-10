"""
CMA-ES Alternating Body–Controller Optimization
------------------------------------------------
Evolves robot bodies and controllers alternately for multiple rounds.
Each round:
  1. Evolve controller for current best body
  2. Evolve body for the latest best controller
"""
# Import standard and third-party libraries
from typing import TYPE_CHECKING, Any
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco as mj
import cma
import wandb

if TYPE_CHECKING:
    from networkx import DiGraph

# ARIEL & local imports

from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json

from baseline import build_initial_population, quick_cull_distance, CULL_THRESHOLD, Individual
from body_evo import random_genotype, decode_to_graph, flatten_genotype, unflatten_genotype, GENOTYPE_SIZE
from utils import init_param_vec, get_in_out_sizes, unpack_weights


# Hyperparameters

POP_SIZE = 16
GENERATIONS = 16
SIGMA_INIT = 0.7
HIDDEN_SIZE = 8

SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
SECONDS = 15

SEED = 42
RNG = np.random.default_rng(SEED)


#Set up saving data

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
    "Hidden Size": HIDDEN_SIZE,
}

# Helper utilities

def expected_ctrl_len(in_size: int, hidden: int, out_size: int) -> int:
    """Total number of weights in 2-hidden-layer MLP."""
    return in_size * hidden + hidden * hidden + hidden * out_size

def coerce_theta_to_shape(theta: np.ndarray,
                          in_size: int,
                          hidden: int,
                          out_size: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Ensure theta length matches expected MLP shape.
    Pads with noise or truncates if lengths differ.
    """
    want = expected_ctrl_len(in_size, hidden, out_size)
    have = len(theta)
    if have == want:
        return theta

    if have > want:
        print(f"[WARN] Controller vector too long ({have}>{want}); truncating.")
        return theta[:want]
    else:
        print(f"[WARN] Controller vector too short ({have}<{want}); padding.")
        pad = rng.normal(0, 0.05, size=(want - have,))
        return np.concatenate([theta, pad])

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc)**2 + (yt - yc)**2 + (zt - zc)**2)


# Set up experiment

def experiment(graph, theta=None, hidden=HIDDEN_SIZE, random=True):
    """Run one rollout for a given body graph and controller."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    core = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    input_size = data.qpos.size + data.qvel.size
    output_size = model.nu

    if random or theta is None:
        w1 = RNG.normal(0, 0.5, size=(input_size, hidden))
        w2 = RNG.normal(0, 0.5, size=(hidden, hidden))
        w3 = RNG.normal(0, 0.5, size=(hidden, output_size))
    else:
        theta = coerce_theta_to_shape(theta, input_size, hidden, output_size, RNG)
        w1, w2, w3 = unpack_weights(theta, input_size, output_size, hidden=hidden)

    def nn_controller(m: mj.MjModel, d: mj.MjData):
        obs = np.concatenate([d.qpos, d.qvel])
        h1 = np.tanh(obs @ w1)
        h2 = np.tanh(h1 @ w2)
        out = np.tanh(h2 @ w3)
        return out * (np.pi / 2)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    simple_runner(model, data, duration=SECONDS)
    return fitness_function(tracker.history["xpos"][0])


# CMA: evolve body

def find_best_body(controller_vec=None, round_idx=0):
    """Evolve the body, optionally with a fixed controller vector."""
    run_name = f"CMA-ES-seed{SEED}-round{round_idx}-find_best_body"
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_name, config=CONFIG)

    init_body_genotype = random_genotype(RNG)
    body_vec = flatten_genotype(init_body_genotype)
    es = cma.CMAEvolutionStrategy(body_vec, SIGMA_INIT, {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9})

    best_body_graph, best_body_vec = None, None
    best_f_overall = -np.inf
    generations = []

    for g in range(GENERATIONS):
        population = es.ask()
        losses, gen_fitness = [], []

        for theta in population:
            body_genotype = unflatten_genotype(theta)
            graph = decode_to_graph(body_genotype)
            f = experiment(graph, theta=controller_vec, random=(controller_vec is None))
            losses.append(-f)
            gen_fitness.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_body_vec = theta.copy()
                best_body_graph = graph

        es.tell(population, losses)
        best_f_in_gen = max(gen_fitness)
        generations.append({"gen": g, "best_f_in_gen": best_f_in_gen, "best_f_overall": best_f_overall})
        run.log({
            "phase": "body", "round": round_idx, "gen": g,
            "best_f_in_gen": best_f_in_gen, "best_f_overall": best_f_overall,
        }, step=g)

    if best_body_graph is not None:
        save_graph_as_json(best_body_graph, DATA / f"best_body_round{round_idx}.json")
        np.save(DATA / f"best_body_vec_round{round_idx}.npy", best_body_vec)

    Path("wandb_artifacts").mkdir(exist_ok=True)
    pd.DataFrame(generations).to_parquet(f"wandb_artifacts/{run_name}_raw.parquet", index=False)
    run.finish()
    return best_body_graph, best_body_vec, best_f_overall


# CMA: evolve controller

def find_best_controller(body_graph, round_idx=0):
    """Evolve the controller for the given body."""
    run_name = f"CMA-ES-seed{SEED}-round{round_idx}-find_best_ctrl"
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_name, config=CONFIG)

    in_size, out_size = get_in_out_sizes(body_graph)
    ctrl_vec = init_param_vec(RNG, in_size, HIDDEN_SIZE, out_size)
    es = cma.CMAEvolutionStrategy(ctrl_vec, SIGMA_INIT, {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9})

    best_ctrl_vec, best_f_overall = None, -np.inf
    generations = []

    for g in range(GENERATIONS):
        population = es.ask()
        losses, gen_fitness = [], []

        for theta in population:
            f = experiment(body_graph, theta=theta, random=False)
            losses.append(-f)
            gen_fitness.append(f)
            if f > best_f_overall:
                best_f_overall = f
                best_ctrl_vec = theta.copy()

        es.tell(population, losses)
        best_f_in_gen = max(gen_fitness)
        generations.append({"gen": g, "best_f_in_gen": best_f_in_gen, "best_f_overall": best_f_overall})
        run.log({
            "phase": "controller", "round": round_idx, "gen": g,
            "best_f_in_gen": best_f_in_gen, "best_f_overall": best_f_overall,
        }, step=g)

    if best_ctrl_vec is not None:
        np.save(DATA / f"best_ctrl_vec_round{round_idx}.npy", best_ctrl_vec)

    Path("wandb_artifacts").mkdir(exist_ok=True)
    pd.DataFrame(generations).to_parquet(f"wandb_artifacts/{run_name}_raw.parquet", index=False)
    run.finish()
    return best_ctrl_vec, best_f_overall


# Alternating main loop

def alternating_main(rounds=3):
    """Iteratively alternate between body and controller evolution."""
    print(f"Starting alternating optimization for {rounds} rounds.")
    best_body_graph, best_body_vec, f_body = find_best_body(None, round_idx=0)
    best_ctrl_vec, f_ctrl = None, -np.inf

    for r in range(1, rounds + 1):
        print(f"\n=== Round {r} ===")
        best_ctrl_vec, f_ctrl = find_best_controller(best_body_graph, round_idx=r)
        print(f"[Round {r}] Controller evolved (fitness: {f_ctrl:.3f})")
        best_body_graph, best_body_vec, f_body = find_best_body(controller_vec=best_ctrl_vec, round_idx=r)
        print(f"[Round {r}] Body evolved (fitness: {f_body:.3f})")

    print("\nOptimization complete")
    print(f"Final body fitness: {f_body:.3f}")
    print(f"Final controller fitness: {f_ctrl:.3f}")
    save_graph_as_json(best_body_graph, DATA / "final_best_body.json")
    np.save(DATA / "final_best_ctrl.npy", best_ctrl_vec)


if __name__ == "__main__":
    alternating_main(rounds=4)

