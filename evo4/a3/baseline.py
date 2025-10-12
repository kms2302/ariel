# examples/a3/baseline.py
"""
Baseline population builder with robust culling.

- Generates random body genotypes
- Decodes to graphs
- Culls individuals that do not move at least CULL_THRESHOLD in a short rollout
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from .body_evo import random_genotype, decode_to_graph, Genotype
from .culling import cull_one

if TYPE_CHECKING:
    from networkx import DiGraph

# --- culling settings ---
CULL_SECONDS    = 2.0    # short, fast
CULL_THRESHOLD  = 0.05   # meters in XY (tune 0.03–0.08)
MAX_ATTEMPTS_FACTOR = 12 # up to pop_size * this many tries

@dataclass
class Individual:
    genotype: Genotype
    robot_graph: "DiGraph[Any]" | None = None
    fitness: float | None = None


def build_initial_population(pop_size: int, rng: np.random.Generator):
    """
    Keep sampling random genotypes until we get 'pop_size' individuals
    that pass the culling threshold. Gives up after pop_size*MAX_ATTEMPTS_FACTOR tries.
    """
    population: list[Individual] = []
    attempts = 0
    max_attempts = pop_size * MAX_ATTEMPTS_FACTOR

    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1

        geno  = random_genotype(rng)
        graph = decode_to_graph(geno)

        disp_m, alive = cull_one(graph, rng, seconds=CULL_SECONDS)
        if alive and disp_m >= CULL_THRESHOLD:
            population.append(Individual(genotype=geno, robot_graph=graph))

    return population
