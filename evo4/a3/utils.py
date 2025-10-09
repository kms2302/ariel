import mujoco as mj
import numpy as np
import math
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# Controller setup: The idea is each joint = A_i * sin(2π f t + φ_i) where
#   - A_i = amplitude of joint i  (We evolve this)
#   - φ_i = phase of joint i      (We evolve this)
#   - f = One shared frequency for all joints (We evolve one value)

# Allowing up to 90 degrees per joint (big enough to push ground)
AMP_MAX = math.pi / 2

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]

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

def cpg_callback(model, data, *, theta):
    """
    ARIEL Central Pattern Generator (CPG) Controller callback.
    Computes joint commands for the current MuJoCo time using the evolved CPG.

    Unpacks the CPG parameters (A, f, phi) from the theta vector.
    """
    t = float(data.time)
    A, f, phi = unpack_params(np.asarray(theta, dtype=np.float64))
    u = A * np.sin(2.0 * math.pi * f * t + phi)
    return np.clip(u, -math.pi/2, math.pi/2)

def rollout_fitness(theta):
    """
    Run one rollout in MuJoCo with ARIEL's Controller class
    and return how far the gecko's core moved in the XY plane (meters).
    """
    # Building a fresh simulation world + robot
    model, data, core = make_model_and_data()

    # Recording the starting XY position of the gecko core
    start_xy = core.xpos[:2].copy()

    # Hooking ARIEL's Controller into MuJoCo 
    # Creating a Controller object that will call our custom cpg_callback at each step
    ctrl = Controller(controller_callback_function=cpg_callback, tracker=None)

    # Saving the old control callback (in case something else set it earlier)
    old_cb = mj.get_mjcb_control()

    # Telling MuJoCo to use our controller: every mj_step, call ctrl.set_control()
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, theta=theta))

    try:
        # Running the simulation for a fixed number of timesteps
        for _ in range(TIMESTEPS):
            mj.mj_step(model, data)  # physics advances, controller callback is invoked automatically
    finally:
        # Restoring the old control callback after rollout
        mj.set_mjcb_control(old_cb)

    # Recording the final XY position
    end_xy = core.xpos[:2].copy()

    # Fitness is the euclidean distance traveled in XY plane 
    return float(np.linalg.norm(end_xy - start_xy))

def get_in_out_sizes(robot_graph):
    # Resetting MuJoCo control callback (this is important to avoid leftover controllers)
    mj.set_mjcb_control(None)

    # Creating the environment
    world = OlympicArena()

    # Turning the body graph into a MuJoCo spec and spawn it into the world
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    # Compiling the MuJoCo model and data
    model = world.spec.compile()
    data = mj.MjData(model)

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

    return input_size, output_size

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
    x0 = pack_weights(w1, w2, w3)

    return x0

def init_pop_vec(rng, pop_size=None):
    """
    Initialize parameters for a population.

    Parameters:
    - rng: Random number generator.
    - pop_size: The size of the population. Default is None.

    Returns:
    - A population of parameter vectors. 
      Population shape: (POP_SIZE, 2*NUM_JOINTS+1).
    """
    assert pop_size, "Pass a pop_size value or use init_param_vec for individual"

    # Initialize a population of parameter vectors
    A0 = rng.normal(0.0, 0.2, (pop_size, NUM_JOINTS))
    phi0 = rng.uniform(0.0, 2.0 * math.pi, (pop_size, NUM_JOINTS))
    f0 = rng.normal(0.0, 0.3, (pop_size, 1))
    x0 = np.concatenate([A0, phi0, f0], axis=1)

    return x0  

def moving_average(x, w=5):
    """Centered moving average with window w."""
    return np.convolve(x, np.ones(w)/w, mode="valid")
