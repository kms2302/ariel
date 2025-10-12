# Standard library
from typing import Any, Tuple
from pathlib import Path

# Third-party
import cma
import numpy as np
import pandas as pd
import wandb
import mujoco as mj
import numpy.typing as npt

# ARIEL modules
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

"""
CO-EVOLUTION of a robot's body and its brain  (fixed-length genome)

- BODY GENOME: 3 probability vectors (type/conn/rot), each length 64
- BRAIN GENOME: large pool of MLP weights. For a given body we SLICE only as
  many weights as needed (I*H + H*H + H*O). Remaining genes are unused.
"""

# ---------------------------- Genome sizes ----------------------------
GENOTYPE_SIZE     = 64
LEN_BODY_GENOME   = 3 * GENOTYPE_SIZE  # 192

# Brain sizing policy (fixed upper bound)
I_MAX        = 35
O_MAX        = 30
HIDDEN_SIZE  = 8
LEN_BRAIN_GENOME = (I_MAX * HIDDEN_SIZE) + (HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * O_MAX)  # 584

LEN_TOTAL_GENOME = LEN_BODY_GENOME + LEN_BRAIN_GENOME  # 776

# ---------------------------- Hyperparams -----------------------------
POP_SIZE    = 16
GENERATIONS = 1024
SIGMA_INIT  = 0.7
SECONDS     = 15.0

# curriculum thresholds for longer rollouts
FIRST_FITNESS_THRESHOLD  = -4.8
SECOND_FITNESS_THRESHOLD = -4.3

SPAWN_POS = [-0.8, 0.0, 0.30]
NUM_OF_MODULES = 30

# W&B
ENTITY  = "evo4"
PROJECT = "assignment3"
CONFIG  = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
    "Hidden Size": HIDDEN_SIZE,
}

# RNG
SEED = 42
RNG  = np.random.default_rng(SEED)

# Output dirs
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA_DIR = Path("__data__") / SCRIPT_NAME
WBART    = Path("wandb_artifacts")
DATA_DIR.mkdir(parents=True, exist_ok=True)
WBART.mkdir(parents=True, exist_ok=True)

# Goal
TARGET_POSITION = np.array([5.42, 0.0, 0.5], dtype=float)  # finish x a bit beyond 5.4


# ---------------------------- Fitness ---------------------------------
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    """
    Return negative Euclidean distance from final 'core' position to target.
    (CMA-ES minimizes, so less negative == better/closer.)
    """
    p = np.array(history[-1], dtype=float)
    return -float(np.linalg.norm(TARGET_POSITION - p))


# ---------------------- Genome creation (random) ----------------------
def make_random_body_genome() -> np.ndarray:
    """
    Body genome are PROBABILITIES in [0,1] for NDE (type/conn/rot).
    """
    return RNG.random(size=LEN_BODY_GENOME, dtype=float)

def make_random_brain_genome() -> np.ndarray:
    """
    Brain genome is a pool of MLP weights. Some tail will be unused depending on body I/O.
    """
    return RNG.normal(loc=0.0, scale=0.3, size=LEN_BRAIN_GENOME)

def make_random_genome() -> np.ndarray:
    body  = make_random_body_genome()
    brain = make_random_brain_genome()
    return np.concatenate([body, brain]).astype(np.float64)


# --------------------------- Decoding (BODY) --------------------------
def decode_body_genome(genome: np.ndarray):
    """
    genome[0:192] -> (type, conn, rot) probabilities -> HPD graph
    """
    g = np.asarray(genome, dtype=np.float64).ravel()
    body = g[:LEN_BODY_GENOME]

    type_p = np.clip(body[0:GENOTYPE_SIZE],              0.0, 1.0).astype(np.float32)
    conn_p = np.clip(body[GENOTYPE_SIZE:2*GENOTYPE_SIZE],0.0, 1.0).astype(np.float32)
    rot_p  = np.clip(body[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE], 0.0, 1.0).astype(np.float32)

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward([type_p, conn_p, rot_p])

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    return robot_graph


def get_io_sizes_for_graph(robot_graph) -> Tuple[int, int, mj.MjModel, mj.MjData, Tracker]:
    """
    Compile the body in the OlympicArena and return (input_size, output_size, model, data, tracker).
    Input size = len(qpos)+len(qvel); Output size = model.nu
    """
    mj.set_mjcb_control(None)
    world = OlympicArena()
    core  = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, position=SPAWN_POS)

    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)
    return input_size, output_size, model, data, tracker


# --------------------------- Decoding (BRAIN) -------------------------
def get_len_required(input_size: int, output_size: int) -> Tuple[int, int, int, int]:
    l1 = input_size * HIDDEN_SIZE
    l2 = HIDDEN_SIZE * HIDDEN_SIZE
    l3 = HIDDEN_SIZE * output_size
    return (l1 + l2 + l3), l1, l2, l3


def decode_brain_genome(robot_graph, genome: np.ndarray):
    """
    DIRECT SLICE MAPPING:
      Use the actual (input_size, output_size) of THIS body to compute
      how many weights we need, then slice that many from the brain pool.
    """
    input_size, output_size, model, data, tracker = get_io_sizes_for_graph(robot_graph)
    len_required, l1, l2, l3 = get_len_required(input_size, output_size)

    # brain pool starts after body segment
    pool = np.asarray(genome, dtype=np.float64).ravel()[LEN_BODY_GENOME:]
    if pool.size < len_required:
        # pad if ever needed (shouldn't happen with our I_MAX/O_MAX)
        pool = np.pad(pool, (0, len_required - pool.size))

    W1 = pool[0:l1].reshape((input_size, HIDDEN_SIZE))
    W2 = pool[l1:l1+l2].reshape((HIDDEN_SIZE, HIDDEN_SIZE))
    W3 = pool[l1+l2:l1+l2+l3].reshape((HIDDEN_SIZE, output_size))

    return W1, W2, W3, model, data, tracker


def decode_genome(genome: np.ndarray):
    """
    Full decode → (controller, model, data, graph, tracker).
    Controller uses tanh MLP with weights (W1,W2,W3) and input = [qpos, qvel].
    """
    robot_graph = decode_body_genome(genome)
    W1, W2, W3, model, data, tracker = decode_brain_genome(robot_graph, genome)

    def nn_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        obs = np.concatenate([d.qpos, d.qvel])             # <<< IMPORTANT: use qpos + qvel
        h1  = np.tanh(obs @ W1)
        h2  = np.tanh(h1 @ W2)
        out = np.tanh(h2 @ W3)
        return np.clip(out, -np.pi/2, np.pi/2)             # keep outputs safe

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    return ctrl, model, data, robot_graph, tracker


# ------------------------------ Main ----------------------------------
def main() -> None:
    run_name = f"CMA-ES-seed{SEED}"
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_name, config=CONFIG)

    x0 = make_random_genome()

    es = cma.CMAEvolutionStrategy(
        x0, SIGMA_INIT, {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9}
    )

    generations = []
    best_f_overall = -np.inf
    best_body_graph = None
    seconds = SECONDS

    for g in range(GENERATIONS):
        population  = es.ask()
        losses      = []
        gen_fitness = []

        for cand in population:
            ctrl, model, data, robot_graph, tracker = decode_genome(cand)

            # Install controller and run
            old = mj.get_mjcb_control()
            mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
            try:
                # dynamic sim length curriculum
                if best_f_overall > SECOND_FITNESS_THRESHOLD and seconds < 160:
                    seconds = 160
                elif best_f_overall > FIRST_FITNESS_THRESHOLD and seconds < 60:
                    seconds = 60

                simple_runner(model, data, duration=seconds)
            finally:
                mj.set_mjcb_control(old)

            f = fitness_function(tracker.history["xpos"][0])
            losses.append(-f)
            gen_fitness.append(f)

            if f > best_f_overall:
                best_f_overall = f
                best_body_graph = robot_graph

        es.tell(population, losses)

        best_f_in_gen = float(np.max(gen_fitness))
        generations.append({
            "gen": g,
            "seconds": seconds,
            "best_f_in_gen": best_f_in_gen,
            "best_f_overall": float(best_f_overall),
        })

        run.log({
            "gen": g,
            "seconds": seconds,
            "best_f_in_gen": best_f_in_gen,
            "best_f_overall": float(best_f_overall),
        }, step=g)

    if best_body_graph is not None:
        save_graph_as_json(best_body_graph, DATA_DIR / f"best_body_{run_name}.json")

    df = pd.DataFrame(generations)
    path = WBART / f"{run_name}_raw.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # no pyarrow/fastparquet? fall back to csv
        path = WBART / f"{run_name}_raw.csv"
        df.to_csv(path, index=False)

    # log artifact
    artifact = wandb.Artifact(
        name=f"{run_name}-raw-data",
        type="raw_data",
        metadata={"generations": GENERATIONS, "rows": len(df)},
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
