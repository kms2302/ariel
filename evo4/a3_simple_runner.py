### ---- REPLACES OUR MJ STEPS WITH SIMPLE RUNNER ----
from typing import Any
import numpy as np
import networkx as nx

from examples.A3_template import (
    NeuralDevelopmentalEncoding,
    HighProbabilityDecoder,
    construct_mjspec_from_graph,
    save_graph_as_json,
    Tracker,
    Controller,
    experiment,
    RNG,
    nn_controller,
    DATA,
    mj,
)


def run_simple_experiment(genotype: list[np.ndarray] | None = None) -> None:
    """
    Build a robot with NDE -> decode to graph -> construct body -> run Mujoco
    in headless 'simple' mode. This function does not return a fitness; it
    just executes a simulation.
    """
    num_modules = 20
    genotype_size = 64

    # Use provided genotype or create a random one
    if genotype is None:
        type_p_genes = RNG.random(genotype_size).astype(np.float32)
        conn_p_genes = RNG.random(genotype_size).astype(np.float32)
        rot_p_genes  = RNG.random(genotype_size).astype(np.float32)
        genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    # Developmental encoding -> probability matrices
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward(genotype)  # (type, connect, rotate)

    # Decode to a directed graph
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph: nx.DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    # Save robot structure
    save_graph_as_json(robot_graph, DATA / "robot_graph.json")

    # Construct Mujoco body
    core = construct_mjspec_from_graph(robot_graph)

    # Track the "core" geom
    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )

    # Simple NN controller from the template
    ctrl = Controller(
        controller_callback_function=nn_controller,
        tracker=tracker,
    )

    # Run the simulation with simple_runner
    experiment(robot=core, controller=ctrl, mode="simple")


if __name__ == "__main__":
    run_simple_experiment()