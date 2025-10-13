"""Robot Olympics Competition template code."""

# Standard library
from pathlib import Path
from typing import Any

# Thirdparty
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    load_graph_from_json,
)
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
TIME_OUT = 500

# Local scripts
from plot_function import show_xpos_history


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = TIME_OUT,
) -> None:
    """Run the simulation with custom controller."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena(
        load_precompiled=False,
    )

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

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
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),  # pyright: ignore[reportUnknownLambdaType]
    )

    # ------------------------------------------------------------------ #

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
    # ==================================================================== #


def main() -> None:
    """Entry point."""
    # Load your robot graph here
    run = "20251013T105557"
    path_to_graph = CWD / f"best_body_robot_graph_{run}.json"
    robot_graph = load_graph_from_json(path_to_graph)

    # Load your weights here
    path_to_w1 = CWD / f"best_brain_w1_{run}.npy"
    path_to_w2 = CWD / f"best_brain_w2_{run}.npy"
    path_to_w3 = CWD / f"best_brain_w3_{run}.npy"

    w1 = np.load(path_to_w1)
    w2 = np.load(path_to_w2)
    w3 = np.load(path_to_w3)

    def nn_controller(
		model: mj.MjModel,
		data: mj.MjData,
	) -> npt.NDArray[np.float64]:
		# Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

		# Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, w1))
        layer2 = np.tanh(np.dot(layer1, w2))
        outputs = np.tanh(np.dot(layer2, w3))

		# Scale the outputs
        return outputs * np.pi

    # Construct the robot from the graph
    core = construct_mjspec_from_graph(robot_graph)

    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=nn_controller,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl)

    # Plot the x-position history
    show_xpos_history(
        tracker.history["xpos"][0],
        spawn_position=SPAWN_POS,
        target_position=TARGET_POSITION,
    )

    # Calculate and print the fitness of your robot
    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Fitness of generated robot: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()
