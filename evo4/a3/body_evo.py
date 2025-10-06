"""
Functions to generate random genotypes and decode them into robot body graphs.
This file makes the robot body evolvable.
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder

# Only used for type hints (so we don’t import networkx at runtime unnecessarily)
if TYPE_CHECKING:
    from networkx import DiGraph

# GLOBAL PARAMETERS:
NUM_OF_MODULES = 30    # maximum number of modules a robot can have
GENOTYPE_SIZE = 64     # length of each probability vector

# DATA STRUCTURE FOR A GENOTYPE
class Genotype:
    """
    A genotype encodes the blueprint of a robot body.
    It is represented by 3 probability vectors:
    - Type_vec: which type of module should appear.
    - Conn_vec: how modules are connected.
    - Rot_vec: how modules are rotated relative to each other.
    These probabilities will later be decoded into an actual robot body graph.
    """
    type_vec: npt.NDArray[np.float32]
    conn_vec: npt.NDArray[np.float32]
    rot_vec: npt.NDArray[np.float32]

    def as_list(self):
        """
        Convenience method: returns the 3 vectors together,
        so they can easily be fed into the NDE decoder.
        """
        return [self.type_vec, self.conn_vec, self.rot_vec]


# RANDOM GENOTYPE GENERATOR
def random_genotype(rng: np.random.Generator, size: int = GENOTYPE_SIZE) -> Genotype:
    """
    Creats a random genotype with 3 probability vectors.
    Each entry is a float in [0,1], sampled uniformly.
    This is the starting point for evolution (random initial bodies).
    """
    return Genotype(
        type_vec=rng.random(size).astype(np.float32),   # random module types
        conn_vec=rng.random(size).astype(np.float32),   # random connections
        rot_vec=rng.random(size).astype(np.float32),    # random rotations
    )


# DECODING GENOTYPE INTO ROBOT BODY
def decode_to_graph(genotype: Genotype) -> "DiGraph[Any]":
    """
    Converts a genotype into a robot body graph.
    Steps:
    1. Using Neural Developmental Encoding (NDE) to transform the 3 probability vectors
       into 3 probability matrices (type, connection, rotation).
    2. Using HighProbabilityDecoder (HPD) to pick the most likely structure from these
       matrices and build an actual graph of modules and joints.
    3. Returns this graph, which fully describes the robot body morphology.
    """
    # Step 1: Run NDE forward pass to produce probability matrices
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward(genotype.as_list())

    # Step 2: Decode those probability matrices into a graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    return hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
