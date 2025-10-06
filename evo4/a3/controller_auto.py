"""
Functions to automatically fit a simple NN controller to any robot body.
The goal here is that every robot, no matter its morphology, gets a controller
with the correct input/output dimensions.
"""

import numpy as np
import mujoco as mj
from ariel.simulation.controllers.controller import Controller


def init_controller(rng, model: mj.MjModel, data: mj.MjData, hidden: int = 8):
    """
    Creates a simple feedforward neural network controller for a given robot.
    
    - Input size: depends on the robot's state (positions + velocities).
    - Output size: depends on how many actuators (joints) the robot has.
    - Hidden size: fixed number of hidden neurons (default = 8).
    
    The weights are initialized randomly each time, which means that in the
    baseline experiments brains are "trained from scratch" for every body.
    """

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

    # Randomly initializing weight matrices for a 3-layer MLP
    # rng.normal(mean, std, size) generates Gaussian-distributed values
    w1 = rng.normal(0, 0.5, size=(input_size, hidden))     # input -> hidden
    w2 = rng.normal(0, 0.5, size=(hidden, hidden))         # hidden -> hidden
    w3 = rng.normal(0, 0.5, size=(hidden, output_size))    # hidden -> output

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

    # Returning the controller function (to be wrapped by ARIEL’s Controller class)
    return nn_controller
