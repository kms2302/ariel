"""
Implements the baseline setup for Assignment 3:
- Evolves robot bodies randomly (brains are randomized fresh each rollout).
- Uses culling: throwing away robots that do not move a minimum distance
  in a short test rollout.
This ensures that our baseline population at least contains "walkable" robots.
"""

from __future__ import annotations

import numpy as np
import mujoco as mj
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# Our own assignment files
from body_evo import random_genotype, decode_to_graph, Genotype
from controller_auto import init_controller

# This is only used for type hints (not the runtime)
if TYPE_CHECKING:
    from networkx import DiGraph

# GLOBAL PARAMETERS:
# This is where the robot is spawned in the OlympicArena
SPAWN_POS = [-0.8, 0, 0.1]

# How long the culling rollout lasts (in seconds of simulation time)
CULL_SECONDS = 2.0

# If a robot moves less than this distance (m) during culling, it is rejected
CULL_THRESHOLD = 0.05


# DATA STRUCTURE FOR INDIVIDUALS:
@dataclass
class Individual:
    """
    Represents a single robot individual in the baseline population.
    - Genotype: the underlying encoding that can be mutated/evolved
    - Robot_graph: the decoded body morphology (graph of modules/joints)
    - Fitness: placeholder for later experiments (it's not used in the baseline)
    """
    genotype: Genotype
    robot_graph: DiGraph[Any] | None = None
    fitness: float | None = None


# CULLING FUNCTION:
def quick_cull_distance(robot_graph: "DiGraph[Any]", rng, seconds=CULL_SECONDS) -> float:
    """
    Running a short simulation of the robot to test if it moves at all.
    Returns the XY displacement (meters).
    If this value is below CULL_THRESHOLD, the robot is considered "dead".
    """

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

    # Setting up a tracker to follow the "core" body part
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # Making a random NN controller that matches the robot’s input/output sizes
    ctrl_fn = init_controller(rng, model, data)

    # Wrapping controller in ARIEL’s Controller class so it can log data
    ctrl = Controller(controller_callback_function=ctrl_fn, tracker=tracker)

    # Telling MuJoCo to call our controller every timestep
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Running the simulation for a short amount of time (fast test, no visualization)
    simple_runner(model, data, duration=seconds)

    # Extracting the robot’s position history from the tracker
    path = np.array(tracker.history["xpos"][0])
    start_xy, end_xy = path[0, :2], path[-1, :2]

    # Return straight-line distance moved in XY plane
    return float(np.linalg.norm(end_xy - start_xy))


# BASELINE POPULATION BUILDER: 
def build_initial_population(pop_size: int, rng: np.random.Generator):
    """
    Creates a baseline population of robots:
    - Starts with random genotypes (random bodies).
    - Decodes them into robot morphologies (graphs).
    - Cull away robots that don’t move at least CULL_THRESHOLD meters.
    - Stop once we have 'pop_size' valid robots (or we give up after many tries).
    """

    population = []   # the list we will fill with Individuals
    attempts = 0      # count how many robots we tried to generate

    # Keep going until we have enough robots or we hit the attempt limit
    while len(population) < pop_size and attempts < pop_size * 10:
        attempts += 1

        # Step 1: generating a random genotype (encodes body layout)
        g = random_genotype(rng)

        # Step 2: decoding genotype -> body morphology graph
        graph = decode_to_graph(g)

        # Step 3: run culling rollout to see if it moves at all
        displacement = quick_cull_distance(graph, rng)

        # Step 4: keep only robots that moved far enough
        if displacement >= CULL_THRESHOLD:
            population.append(Individual(genotype=g, robot_graph=graph))

    # Return the list of Individuals (some may be rejected if they failed culling)
    return population
