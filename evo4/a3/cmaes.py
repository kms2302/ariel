# Import standard libraries
from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path
import json

# Import third-party libraries
import wandb
import numpy as np
import cma
import mujoco as mj
import pandas as pd
from networkx.readwrite import json_graph

# Type Aliases and Checking
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]
if TYPE_CHECKING:
    from networkx import DiGraph

# Import local ARIEL modules from CI Group
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    save_graph_as_json,
)

# Import local modules from the same directory as the current file
from baseline import (
    build_initial_population,
    quick_cull_distance,
    CULL_THRESHOLD,
    Individual,
)
from body_evo import (
    random_genotype,
    decode_to_graph,
    flatten_genotype,
    unflatten_genotype,
    GENOTYPE_SIZE,
)
from utils import (
    init_param_vec,
    get_in_out_sizes,
    unpack_weights,
    make_controller,
    controller_len,
)

# Hyperparameters
POP_SIZE = 16
GENERATIONS = 16
SIGMA_INIT = 0.7
HIDDEN_SIZE = 8

# Global constants
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
SECONDS = 15

# Random generator setup
SEED = 42
RNG = np.random.default_rng(SEED)

# Setup for saving data to JSON
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
}

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def experiment(
    graph,
    theta=None,
    hidden=HIDDEN_SIZE,
    random=True,
):
    """Run an experiment with a single robot"""
    # Resetting MuJoCo control callback (this is important to avoid leftover controllers)
    mj.set_mjcb_control(None)

    # Creating the environment
    world = OlympicArena()

    # Turning the body graph into a MuJoCo spec and spawn it into the world
    core = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=SPAWN_POS)

    # Compiling the MuJoCo model and data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Setting up a tracker to follow the "core" body part
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # The input size is based on the robot's own state:
    # - qpos = how many joint angles/positions the robot has
    # - qvel = how many joint velocities it has
    # Different robots have different numbers of joints,
    # so this makes the controller automatically match each new body.
    input_size = data.qpos.size + data.qvel.size

    # The output size equals the number of actuators (motors) in the robot.
    # Each robot body can have a different number of joints to control,
    # so this makes sure the controller always produces the right amount of outputs.
    output_size = model.nu

    if random:
        # Randomly initializing weight matrices for a 3-layer MLP
        # rng.normal(mean, std, size) generates Gaussian-distributed values
        w1 = RNG.normal(0, 0.5, size=(input_size, hidden))     # input -> hidden
        w2 = RNG.normal(0, 0.5, size=(hidden, hidden))         # hidden -> hidden
        w3 = RNG.normal(0, 0.5, size=(hidden, output_size))    # hidden -> output
    else:
        w1, w2, w3 = unpack_weights(theta, input_size, output_size)

    # Defining the actual controller callback that MuJoCo will call every timestep
    def nn_controller(m: mj.MjModel, d: mj.MjData):
        # Creates the input for the neural network by joining together:
        # - qpos (all joint positions/angles)
        # - qvel (all joint velocities)
        # This gives the network the full current state of the robot.
        obs = np.concatenate([d.qpos, d.qvel])

        # Forward pass through the neural network (tanh activations)
        h1 = np.tanh(obs @ w1)     # input layer -> hidden layer 1
        h2 = np.tanh(h1 @ w2)      # hidden layer 1 -> hidden layer 2
        out = np.tanh(h2 @ w3)     # hidden layer 2 -> output layer

        # Scaling outputs to joint limits (±π/2 radians)
        return out * (np.pi / 2)

    ctrl = Controller(
        controller_callback_function=nn_controller,
        tracker=tracker,
    )

    # Telling MuJoCo to call our controller every timestep
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Running the simulation for a short amount of time (fast test, no visualization)
    simple_runner(model, data, duration=SECONDS)
    f = fitness_function(tracker.history["xpos"][0])  # Displacement (bigger is better)

    return f

def find_best_body():
    # Start a new wandb run to track this script.
    run_name = f"CMA-ES-seed{SEED}-find_best_body"
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        config=CONFIG,
    )

    init_body_genotype = random_genotype(RNG)
    init_body_graph = decode_to_graph(init_body_genotype)
    init_input_size, init_output_size = get_in_out_sizes(init_body_graph)

    body_vec = flatten_genotype(init_body_genotype)  # length = 3*GENOTYPE_SIZE
    ctrl_vec = init_param_vec(RNG, init_input_size, HIDDEN_SIZE, init_output_size)

    x0 = body_vec

    # CMA for evolving this body (find best body for random brains)
    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA_INIT,
        {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9},  # quiet logs
    )
    generations = []
    best_per_gen = []
    best_f_overall = -np.inf  # best fitness seen overall

    for g in range(GENERATIONS):
        population = es.ask()
        losses = []
        gen_fitness = []

        for theta in population:
            body_genotype = unflatten_genotype(theta)
            graph = decode_to_graph(body_genotype)

            # Compute and store the robot's fitness
            f = experiment(graph, random=True)  # Displacement (bigger is better)
            losses.append(-f)  # Negate. Lower is better for CMA
            gen_fitness.append(f)

            if f > best_f_overall:
                best_f_overall = f
                best_body_vec   = body_vec.copy()      # store flat vector
                best_body_graph = graph                # store decoded graph

        es.tell(solutions=population, function_values=losses)

        # For plotting: keeping the best (i.e., max displacement) this generation
        best_f_in_gen = max(gen_fitness)
        best_per_gen.append(best_f_in_gen)

        # Log this gen (i.e., step) to Weights & Biases
        run.log({
            "gen": g,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        }, step=g)

        # Append raw rows for this generation
        generations.append({
            "gen": g,
            "gen_fitness": gen_fitness,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        })

    if best_body_graph is not None:
        # Save the graph as JSON for later inspection and dump the flat vector (useful for re‑loading)
        save_graph_as_json(best_body_graph, DATA / f"best_body_{run_name}.json")
        np.save(DATA / f"best_body_vec_{run_name}.npy", best_body_vec)
    
    # End of run: create a DataFrame and write to Parquet (or CSV)
    out_dir = Path("wandb_artifacts")
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / f"{run_name}_raw.parquet"
    df = pd.DataFrame(generations)
    df.to_parquet(file_path, index=False)

    # Create an artifact, add the file, and log it
    artifact = wandb.Artifact(
        name=f"{run_name}-raw-data",
        type="raw_data",
        metadata={"generations": GENERATIONS, "num_rows": len(df)}
    )
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)

    # Finish the run and upload any remaining data.
    run.finish()

    return best_body_graph


def main() -> None:
    best_body_graph = find_best_body()

    # Start a new wandb run to track this script.
    run_name = f"CMA-ES-seed{SEED}-find_best_ctrl"
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        config=CONFIG,
    )

    init_input_size, init_output_size = get_in_out_sizes(best_body_graph)
    ctrl_vec = init_param_vec(RNG, init_input_size, HIDDEN_SIZE, init_output_size)

    x0 = ctrl_vec

    # CMA for evolving the brain (find best brain for the body that is best for random brains)
    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA_INIT,
        {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9},  # quiet logs
    )
    generations = []
    best_per_gen = []
    best_f_overall = -np.inf  # best fitness seen overall

    for g in range(GENERATIONS):
        population = es.ask()
        losses = []
        gen_fitness = []

        for theta in population:
            w1, w2, w3 = unpack_weights(theta, init_input_size, init_output_size)

            # Compute and store the robot's fitness
            f = experiment(best_body_graph, theta=theta, random=False)  # Displacement (bigger is better)
            losses.append(-f)  # Negate. Lower is better for CMA
            gen_fitness.append(f)

            if f > best_f_overall:
                best_f_overall = f
                best_ctrl_vec   = theta.copy()      # store flat vector
                
        es.tell(solutions=population, function_values=losses)

        # For plotting: keeping the best (i.e., max displacement) this generation
        best_f_in_gen = max(gen_fitness)
        best_per_gen.append(best_f_in_gen)

        # Log this gen (i.e., step) to Weights & Biases
        run.log({
            "gen": g,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        }, step=g)

        # Append raw rows for this generation
        generations.append({
            "gen": g,
            "gen_fitness": gen_fitness,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        })

    if best_ctrl_vec is not None:
        np.save(DATA / f"best_ctrl_vec_{run_name}.npy", best_ctrl_vec)
    
    # End of run: create a DataFrame and write to Parquet (or CSV)
    out_dir = Path("wandb_artifacts")
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / f"{run_name}_raw.parquet"
    df = pd.DataFrame(generations)
    df.to_parquet(file_path, index=False)

    # Create an artifact, add the file, and log it
    artifact = wandb.Artifact(
        name=f"{run_name}-raw-data",
        type="raw_data",
        metadata={"generations": GENERATIONS, "num_rows": len(df)}
    )
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)

    # Finish the run and upload any remaining data.
    run.finish()

if __name__ == "__main__":
    main()
