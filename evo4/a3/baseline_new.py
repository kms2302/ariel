# examples/a3/baseline_new.py
# 
# Random-search baseline for assignment A3. It's aligned with our cmaes_new.py file.
#
# What does this script:
# - It uses the exact same genome layout as our CMA-ES setup:
#   * BODY: 3 probability vectors of length 64 (type / connection / rotation)
#   * BRAIN: a fixed-size pool of MLP weights (we slice only what the body needs)
# - Every generation, we just sample a new random population and evaluate them.
#   There is no selection, no mutation and no adaptation: this is pure random search.
# - For each robot we:
#   1) decode the body -> make a MuJoCo graph/model
#   2) slice & reshape brain weights -> build controller
#   3) simulate for a fixed number of seconds
#   4) Scoring the robot the same way as in CMA-ES by taking the negative distance to the target (XY plane)
#    and subtract a small penalty if it didn’t reach the target.
# - We log a simple CSV with best-of-generation and best-overall.
# - We also save the single best baseline robot (body JSON + W1/W2/W3) so we can replay it.

from pathlib import Path
import time
import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd
import mujoco as mj
import numpy.typing as npt

# ARIEL imports (the same as we use in cmaes_new.py)
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# Constants (they match with our cmaes_new.py)

# Body genome = 3 * 64 probability-like values
GENOTYPE_SIZE = 64
LEN_BODY_GENOME = 3 * GENOTYPE_SIZE  # Which is 192

# Controller (brain) pool dimensions (we slice to actual I/O sizes per body)
I_MAX = 35
O_MAX = 30
HIDDEN_SIZE = 8
LEN_BRAIN_GENOME = I_MAX * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * O_MAX  # 584
LEN_TOTAL_GENOME = LEN_BODY_GENOME + LEN_BRAIN_GENOME

NUM_OF_MODULES = 30  # Number of modules for the high-prob decoder

# Task: making the robot reach the target
# For the distance we only look at x and y (see fitness function below)
TARGET_POSITION = np.array([5.0, 0.0, 0.5], dtype=float)

# Spawn options: base start + slightly advanced starts with penalties
SPAWN_POS        = np.array([-0.8, 0.0, 0.1])
RUGGED_PENALTY   = 0.3
INCLINED_PENALTY = 1.3
RUGGED_POS       = SPAWN_POS + np.array([RUGGED_PENALTY, 0, 0])
INCLINED_POS     = SPAWN_POS + np.array([INCLINED_PENALTY, 0, 0])

# We randomly pick one of these for each individual
# The penalty only applies if we didn’t exactly reach the target
SPAWN_OPTIONS = [
    {"pos": SPAWN_POS,    "penalty": 0.0},
    {"pos": RUGGED_POS,   "penalty": RUGGED_PENALTY},
    {"pos": INCLINED_POS, "penalty": INCLINED_PENALTY},
]

# Genome helper functions

def make_random_body_genome(rng: np.random.Generator) -> np.ndarray:
    """
    Body “genes” for the NDE (Neural Developmental Encoding).
    We reuse the same range as our CMA-ES code to keep the baseline comparable.
    """
    return rng.uniform(low=-100, high=100, size=LEN_BODY_GENOME)

def make_random_brain_genome(rng: np.random.Generator) -> np.ndarray:
    """
    Brain weight pool. Later we slice/reshape to W1, W2, W3 depending on
    the actual input/output sizes of the compiled body.
    """
    return rng.normal(loc=0.0138, scale=0.5, size=LEN_BRAIN_GENOME)

def make_random_genome(rng: np.random.Generator) -> np.ndarray:
    """Full genome = body segment + brain segment."""
    return np.concatenate([make_random_body_genome(rng), make_random_brain_genome(rng)])

def get_len_required(input_size: int, output_size: int):
    """
    Given the body’s actual input/output sizes, computing how many weights we need
    for a simple 2-hidden-layer MLP (HIDDEN_SIZE neurons per hidden layer).
    """
    l1 = input_size * HIDDEN_SIZE
    l2 = HIDDEN_SIZE * HIDDEN_SIZE
    l3 = HIDDEN_SIZE * output_size
    return l1 + l2 + l3, l1, l2, l3

# Individual class 

class Individual:
    """
    A single random robot:
    - holds the genome vector
    - decodes body -> builds a MuJoCo model
    - decodes brain -> builds a simple tanh MLP controller
    """

    def __init__(self, genome: npt.NDArray[np.float64], spawn_data: dict):
        self.genome = genome

        # Info about where this robot starts (spawn position + penalty)
        self.spawn_data = spawn_data
        self.spawn_position = spawn_data["pos"].tolist()
        self.spawn_penalty  = float(spawn_data["penalty"])

        # These will be set later when we decode the genome
        self.robot_graph = None
        self.model = None
        self.data = None
        self.tracker = None
        self.controller = None

        # MLP weights
        self.w1 = self.w2 = self.w3 = None

    def decode(self):
        """Full decode pipeline: body -> env -> brain -> controller."""
        self._decode_body()
        self._setup_env()
        self._decode_brain()
        self._make_controller()

    def _decode_body(self):
        """Turning the first 192 genes into probability matrices -> body graph."""
        body = self.genome[:LEN_BODY_GENOME]
        type_p = body[0:GENOTYPE_SIZE]
        conn_p = body[GENOTYPE_SIZE:2*GENOTYPE_SIZE]
        rot_p  = body[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE]

        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p = nde.forward([type_p, conn_p, rot_p])

        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        self.robot_graph = hpd.probability_matrices_to_graph(p[0], p[1], p[2])

    def _setup_env(self):
        """
        Builds the MuJoCo model and data for this body. After that it spawns it in the chosen spot
        and attach a tracker that records (x, y, t) during the rollout.
        """
        # Makes sure no previous controller is still registered
        mj.set_mjcb_control(None)

        world = OlympicArena()
        core  = construct_mjspec_from_graph(self.robot_graph)
        world.spawn(core.spec, position=self.spawn_position)

        self.model = world.spec.compile()
        self.data  = mj.MjData(self.model)
        mj.mj_resetData(self.model, self.data)

        self.tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        self.tracker.setup(world.spec, self.data)

    def _decode_brain(self):
        """
        Slicing the brain pool to the sizes needed by this body, then reshape into W1, W2, W3.
        Inputs = qpos only (matches cmaes_new.py).
        """
        input_size  = len(self.data.qpos)
        output_size = int(self.model.nu)

        need, l1, l2, l3 = get_len_required(input_size, output_size)
        pool = self.genome[LEN_BODY_GENOME:LEN_BODY_GENOME+need]

        self.w1 = pool[0:l1].reshape((input_size, HIDDEN_SIZE))
        self.w2 = pool[l1:l1+l2].reshape((HIDDEN_SIZE, HIDDEN_SIZE))
        self.w3 = pool[l1+l2:l1+l2+l3].reshape((HIDDEN_SIZE, output_size))

    def _make_controller(self):
        """
        # Simple tanh MLP:
        #   qpos -> tanh -> tanh -> tanh
        #   outputs are multiplied so they match the torque range of the robot's motors
        This is intentionally the same as in cmaes_new.py so the baseline is fair.
        """
        W1, W2, W3 = self.w1, self.w2, self.w3

        def nn_controller(m: mj.MjModel, d: mj.MjData):
            x  = d.qpos
            h1 = np.tanh(x @ W1)
            h2 = np.tanh(h1 @ W2)
            out = np.tanh(h2 @ W3)
            return out * np.pi  # keep identical scaling

        self.controller = Controller(controller_callback_function=nn_controller, tracker=self.tracker)

# Fitness

def fitness(history: list[tuple[float,float,float]], penalty: float) -> float:
    """
    Our tracker stores tuples (x, y, t). We measure XY distance to the target (ignore z),
    then take the negative (so closer is better). If we didn’t reach the exact target,
    we subtract a small penalty depending on the spawn section.
    """
    xt, yt, _ = TARGET_POSITION
    xc, yc, _t = history[-1]  # last position/time sample
    dist_xy = float(np.hypot(xt - xc, yt - yc))
    score = -dist_xy
    if score < 0.0:  # Only applies the penalty if we didn't reach the target
        score -= float(penalty)
    return score

# Main 

def main():
    parser = argparse.ArgumentParser(description="Random-search baseline (no learning).")
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--generations", type=int, default=50, help="Number of baseline generations")
    parser.add_argument("--popsize", type=int, default=32, help="Random individuals per generation")
    parser.add_argument("--seconds", type=float, default=15.0, help="Rollout duration (fixed)")
    parser.add_argument("--outdir", type=str, default="__baseline__")
    args = parser.parse_args()

    # To make the results repeatable we set the seed for both numpy and random (for spawn picking)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    now = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"baseline_seed{args.seed}_{now}"
    csv_path = outdir / f"{run_name}.csv"

    best_overall_f = -np.inf
    best_robot: Optional[Individual] = None
    rows = []

    for g in range(args.generations):
        best_f_in_gen = -np.inf
        best_ind_gen: Optional[Individual] = None

        for _ in range(args.popsize):
            # 1) Sampling a fresh random genome + random spawn config
            genome = make_random_genome(rng)
            spawn_data = random.choice(SPAWN_OPTIONS)

            # 2) Building the robot (body + controller)
            ind = Individual(genome, spawn_data)
            ind.decode()

            # 3) Simulating with safe control callback install/restore
            old = mj.get_mjcb_control()
            mj.set_mjcb_control(lambda m, d: ind.controller.set_control(m, d))
            try:
                simple_runner(ind.model, ind.data, duration=args.seconds)
            finally:
                mj.set_mjcb_control(old)

            # 4) We score the robot the same way as in CMA-ES (so results are comparable)
            f = fitness(ind.tracker.history["xpos"][0], ind.spawn_penalty)

            # Tracks the best individual in this generation
            if f > best_f_in_gen:
                best_f_in_gen = f
                best_ind_gen = ind

        # Updates the global best across generations
        if best_ind_gen is not None and best_f_in_gen > best_overall_f:
            best_overall_f = best_f_in_gen
            best_robot = best_ind_gen

        rows.append({
            "gen": g,
            "seconds": float(args.seconds),
            "best_f_in_gen": float(best_f_in_gen),
            "best_f_overall": float(best_overall_f),
        })
        print(f"[BASELINE GEN {g:03d}] best_f_in_gen={best_f_in_gen:.4f}  best_f_overall={best_overall_f:.4f}")

    # Saves the results to a CSV file so we can later plot best_f_overall vs generation
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved baseline curve: {csv_path}")

    # Saves the best robot we found (so we can reload or replay it later if needed)
    if best_robot is not None:
        save_dir = outdir / f"{run_name}_best_robot"
        save_dir.mkdir(exist_ok=True)
        save_graph_as_json(best_robot.robot_graph, save_dir / "baseline_best_body.json")
        np.save(save_dir / "baseline_best_w1.npy", best_robot.w1)
        np.save(save_dir / "baseline_best_w2.npy", best_robot.w2)
        np.save(save_dir / "baseline_best_w3.npy", best_robot.w3)
        print(f"Saved best baseline robot to: {save_dir}")

if __name__ == "__main__":
    main()