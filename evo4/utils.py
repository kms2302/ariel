import mujoco
import numpy as np
import math
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments.boxy_heightmap import BoxyRugged
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Controller setup: The idea is each joint = A_i * sin(2π f t + φ_i) where
#   - A_i = amplitude of joint i  (We evolve this)
#   - φ_i = phase of joint i      (We evolve this)
#   - f = One shared frequency for all joints (We evolve one value)

# Allowing up to 90 degrees per joint (big enough to push ground)
AMP_MAX = math.pi / 2

def make_model_and_data():
    """
    Model setup: Building a fresh world + robot, compiling MuJoCo model and
    return (model, data, core_binding).
    """
    # Clearing any leftover control callback (MuJoCo keeps a global pointer)
    mujoco.set_mjcb_control(None)

    # Creating an environment and robot, then spawn it
    world = BoxyRugged()
    body  = gecko()
    # 0.08: A tiny lift so we don't clip into ground
    world.spawn(body.spec, spawn_position=[0, 0, 0.08])  

    # MuJoCo needs a compiled model and a data object that holds the state
    model = world.spec.compile()
    data  = mujoco.MjData(model)  # type: ignore

    # We need to track the XY of the "core" geom to measure how far we walked
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    core  = [data.bind(geom) for geom in geoms if "core" in geom.name][0]

    return model, data, core

def read_model_meta():
    """
    Metadata lookup: Quick one-time call to learn how many joints we have
    and what the simulation time step in seconds (dt) is.

    Output:
    - model.nu: number of joints (i.e., actuators)
    - model.opt.timestep: MuJoCo dt (i.e., simulation time step in seconds)
    """
    model, _, _ = make_model_and_data()
    return model.nu, model.opt.timestep  # The number of actuators, MuJoCo dt

def unpack_params(theta):
    """
    Parameter unpacking: Turning an unconstrained parameter vector
    into (A, f, phi) with the right bounds.

    theta layout (flat 1D array):
      [A_1 .. A_n,  phi_1 .. phi_n,  f_shared]
       |--- n ---|  |----- n -----|  |   1    |
    """
    assert theta.size == NUM_JOINTS * 2 + 1, "wrong vector size"

    # Splitting the flat vector into blocks
    A_raw   = theta[:NUM_JOINTS]
    phi_raw = theta[NUM_JOINTS:2*NUM_JOINTS]
    f_raw   = theta[-1]  # Just one number

    # Mapping to useful ranges:
    # 1. Amplitudes: (0, AMP_MAX) via sigmoid
    A = AMP_MAX * (1.0 / (1.0 + np.exp(-A_raw)))

    # 2. Phases: wrap everything into [0, 2π)
    phi = (phi_raw % (2.0 * math.pi))

    # 3. Shared frequency: put into [0.8, 2.2] Hz with sigmoid
    f = 0.8 + 1.4 * (1.0 / (1.0 + np.exp(-f_raw)))

    return A, f, phi

def cpg_callback(model, data, *, theta):
    """
    ARIEL Controller callback.
    Computes joint commands for the current MuJoCo time using the evolved CPG.
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
    old_cb = mujoco.get_mjcb_control()

    # Telling MuJoCo to use our controller: every mj_step, call ctrl.set_control()
    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, theta=theta))

    try:
        # Running the simulation for a fixed number of timesteps
        for _ in range(TIMESTEPS):
            mujoco.mj_step(model, data)  # physics advances, controller callback is invoked automatically
    finally:
        # Restoring the old control callback after rollout
        mujoco.set_mjcb_control(old_cb)

    # Recording the final XY position
    end_xy = core.xpos[:2].copy()

    # itness is the euclidean distance traveled in XY plane 
    return float(np.linalg.norm(end_xy - start_xy))

def init_param_vec(rng):
    """
    Initialize parameters for a single individual.

    Parameters:
    - rng: Random number generator.

    Returns:
    - An individual parameter vector. If pop_size is 1, it returns a single
      parameter vector with shape (2*NUM_JOINTS+1,)
    """
    A0 = rng.normal(0.0, 0.2, NUM_JOINTS)
    phi0 = rng.uniform(0.0, 2.0 * math.pi, NUM_JOINTS)
    f0 = rng.normal(0.0, 0.3, 1)
    x0 = np.concatenate([A0, phi0, f0])

    return x0  # Individual: 

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


NUM_JOINTS, DT = read_model_meta()
