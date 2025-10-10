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
from baseline import build_initial_population
from body_evo import random_genotype, decode_to_graph
from utils import (
    init_param_vec,
    get_in_out_sizes,
    unpack_weights,
    make_controller,
)

# Global constants
BODY_POP_SIZE = 8
POP_SIZE = 256
GENERATIONS = 8
SIGMA_INIT = 0.4
HIDDEN_SIZE = 8
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

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

def headless_experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
) -> None:
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    simple_runner(model, data, duration=duration)


def main() -> None:
    # Build initial body population
    body_pop = build_initial_population(pop_size=BODY_POP_SIZE, rng=RNG)

    for body_idx, body in enumerate(body_pop):
        # Start a new wandb run to track this script.
        run_name = f"CMA-ES-seed{SEED}-body{body_idx}"
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=run_name,
            config={**CONFIG, "body_idx": body_idx},
        )

        robot_graph: DiGraph[Any] = body.robot_graph
        input_size, output_size = get_in_out_sizes(robot_graph)
        x0 = init_param_vec(RNG, input_size, HIDDEN_SIZE, output_size)
        save_graph_as_json(
            robot_graph,
            DATA / "robot_graph.json",
        )

        # Setting up CMA-ES
        # Note: CMA-ES MINIMIZES by default, so we'll pass negative fitness values
        es = cma.CMAEvolutionStrategy(
            x0,
            SIGMA_INIT,
            {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9},  # quiet logs
        )

        best_per_gen = []
        best_f_overall = -np.inf
        generations = []

        for g in range(GENERATIONS):
            # ask(): sampling population of candidate controller solutions
            population = es.ask()
            losses = []
            gen_fitness = []

            for theta in population:
                # Setting up a tracker to follow the "core" body part
                tracker = Tracker(
                    mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
                    name_to_bind="core",
                )

                w1, w2, w3 = unpack_weights(theta, input_size, output_size)

                # sanity check
                assert w1.shape[0] == input_size

                ctrl_fn = make_controller(w1, w2, w3, input_size)

                # Simple NN controller to simulate the robot
                ctrl = Controller(
                    controller_callback_function=ctrl_fn,
                    tracker=tracker,
                )

                core = construct_mjspec_from_graph(robot_graph)

                # Run an experiment with a single robot
                headless_experiment(robot=core, controller=ctrl)

                # Compute and store the robot's fitness
                f = fitness_function(tracker.history["xpos"][0])  # Displacement (bigger is better)
                losses.append(-f)  # Negate. Lower is better for CMA
                gen_fitness.append(f)

            # For plotting: keeping the best (i.e., max displacement) this generation
            best_f_in_gen = max(gen_fitness)
            best_per_gen.append(best_f_in_gen)

            # Update overall best across generations
            best_f_overall = max(best_f_overall, best_f_in_gen)

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
