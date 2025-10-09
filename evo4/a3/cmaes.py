# Import standard libraries
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Import third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import wandb
import cma
import pandas as pd

# Import local ARIEL modules
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Same directory: local modules
from utils import (
    cpg_callback,
    init_param_vec,
    get_in_out_sizes,
    unpack_weights,
    make_controller,
)
from params import (
    GENERATIONS,
    POP_SIZE,
    SIGMA_INIT,
    ENTITY,
    PROJECT,
    CONFIG,
)
from baseline import build_initial_population
from body_evo import random_genotype, decode_to_graph, Genotype

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# Random generator setup
SEED = 42
RNG = np.random.default_rng(SEED)

# Data collection setup for saving robot structure
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
GENOTYPE_SIZE = 64  # length of each of the 3 vectors
HIDDEN_SIZE = 8


def gen_rand_robot_body(genotype_size) -> list[np.ndarray]:
    """
    Creates and returns a randomly generated genotype.
    
    Input:
    - genotype_size: length of each of the 3 vectors

    Output:
    - genotype: list of 3 vectors
      - type: ...
      - connect: ...
      - rotate: ...
    """

    # Generate random genes for an individual's genotype
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    # Create the input for the NDE (represents a single individual)
    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]
    return genotype

def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()

def graph_to_json(graph):
    data = json_graph.node_link_data(graph, edges="edges")
    json_string = json.dumps(data, indent=4)
    return json_string

def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Clearing any leftover control callback (MuJoCo keeps a global pointer)
    mj.set_mjcb_control(None)

    # Initialise world and robot body
    world = OlympicArena()
    body = robot

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(body.spec, position=SPAWN_POS)

    # MuJoCo needs a compiled model and a data object that holds the state.
    # These are standard parts of the simulation.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Telling MuJoCo to use our controller: every mj_step, call ctrl.set_control()
    # This is called every time step to get the next action.
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


def main() -> None:
    # Start a new wandb run to track this script.
    run_name = f"CMA-ES-seed{SEED}"
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        config=CONFIG,
    )
    wandb.config.update({
        "Initial Sigma": SIGMA_INIT,
    })

    # Build initial body population
    body_pop_size = 5
    body_pop = build_initial_population(pop_size=body_pop_size, rng=RNG)

    for body_idx, body in enumerate(body_pop):
        # Initialize population of candidate controller solutions
        genotype = random_genotype(rng=RNG, size=GENOTYPE_SIZE)
        robot_graph: DiGraph[Any] = decode_to_graph(genotype)
        input_size, output_size = get_in_out_sizes(robot_graph)
        hidden = HIDDEN_SIZE

        x0 = init_param_vec(RNG, input_size, hidden, output_size)

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
                w1, w2, w3 = unpack_weights(theta, input_size, output_size)

                # sanity check
                assert w1.shape[0] == input_size

                ctrl_fn = make_controller(w1, w2, w3, input_size)

                # Track the "core" geom
                tracker = Tracker(
                    mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
                    name_to_bind="core",
                )

                # Simple NN controller to simulate the robot
                ctrl = Controller(
                    controller_callback_function=ctrl_fn,
                    tracker=tracker,
                )

                core = construct_mjspec_from_graph(robot_graph)

                # Run an experiment with a single robot
                experiment(robot=core, controller=ctrl, mode="simple")

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
                "body_idx": body_idx,
                "gen": g,
                "Best fitness in generation": best_f_in_gen, 
                "Best fitness across generations": best_f_overall,
            }, step=g)

            # Append raw rows for this generation
            generations.append({
                "body_idx": body_idx,
                "body_graph": graph_to_json(robot_graph),
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
