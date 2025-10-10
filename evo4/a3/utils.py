# Import third-party libraries
import mujoco as mj
import numpy as np
import math

# Import local ARIEL modules from CI Group
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# Allowing up to 90 degrees per joint (big enough to push ground)
AMP_MAX = math.pi / 2

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]


def controller_len(input_size: int, hidden_size: int, output_size: int) -> int:
    """Number of scalar parameters required for a 2‑layer NN."""
    return (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)

def pack_weights(w1, w2, w3):
    """Concatenate all weight matrices into a flat vector."""
    return np.concatenate([w1.ravel(), w2.ravel(), w3.ravel()])

def unpack_weights(theta, input_size, output_size, hidden=8):
    """Recover the three matrices from the flat vector."""
    i1 = input_size * hidden
    i2 = i1 + hidden * hidden
    i3 = i2 + hidden * output_size

    w1 = theta[:i1].reshape(input_size, hidden)
    w2 = theta[i1:i2].reshape(hidden, hidden)
    w3 = theta[i2:i3].reshape(hidden, output_size)
    return w1, w2, w3

def make_controller(w1, w2, w3, input_dim):
    def nn_controller(m: mj.MjModel, d: mj.MjData):
        obs = np.concatenate([d.qpos, d.qvel])[:input_dim]
        h1 = np.tanh(obs @ w1)
        h2 = np.tanh(h1 @ w2)
        out = np.tanh(h2 @ w3)
        return out * (np.pi / 2)
    return nn_controller

def get_in_out_sizes(robot_graph):
    # Resetting MuJoCo control callback (this is important to avoid leftover controllers)
    mj.set_mjcb_control(None)

    # Creating the environment
    world = OlympicArena()

    # Turning the body graph into a MuJoCo spec and spawn it into the world
    core = construct_mjspec_from_graph(robot_graph)
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
    init_input_size = data.qpos.size + data.qvel.size

    # The output size equals the number of actuators (motors) in the robot.
    # Each robot body can have a different number of joints to control,
    # so this makes sure the controller always produces the right amount of outputs.
    init_output_size = model.nu

    return init_input_size, init_input_size

def init_param_vec(rng, input_size, hidden, output_size):
    """
    Initialize parameters for a single individual.

    Parameters:
    - rng: Random number generator.

    Returns:
    - An individual parameter vector. If pop_size is 1, it returns a single
      parameter vector
    """
    # Randomly initializing weight matrices for a 3-layer MLP
    # rng.normal(mean, std, size) generates Gaussian-distributed values
    w1 = rng.normal(0, 0.5, size=(input_size, hidden))     # input -> hidden
    w2 = rng.normal(0, 0.5, size=(hidden, hidden))         # hidden -> hidden
    w3 = rng.normal(0, 0.5, size=(hidden, output_size))    # hidden -> output
    init_ctrl_vec = pack_weights(w1, w2, w3)

    return init_ctrl_vec
