# Implements the BODY/brain split and DIRECT SLICE MAPPING (steps 7–10).

from __future__ import annotations
import numpy as np
import mujoco as mj
from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# Body genome constants (3 probability vectors, size=64 each)
GENOTYPE_SIZE = 64
LEN_BODY_GENOME = 3 * GENOTYPE_SIZE   # 192

# Brain sizing policy for a fixed-length CMA-ES genome
HIDDEN_SIZE = 8
I_MAX = 35       # robust upper bound for len(qpos)+len(qvel)
O_MAX = 30       # robust upper bound for nu (actuators)
LEN_BRAIN_GENOME = (I_MAX * HIDDEN_SIZE) + (HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * O_MAX)  # 584

# Final fixed genome length (body + brain)
TOTAL_GENOME_LEN = LEN_BODY_GENOME + LEN_BRAIN_GENOME  # 776


def slice_body_brain(genome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split fixed-length genome into:
      body_part  = first 192 genes
      brain_pool = next 584 genes (unused tail is ignored by design)
    """
    x = np.asarray(genome, dtype=float).ravel()
    if x.size < TOTAL_GENOME_LEN:
        x = np.pad(x, (0, TOTAL_GENOME_LEN - x.size))
    body = x[:LEN_BODY_GENOME]
    brain_pool = x[LEN_BODY_GENOME:LEN_BODY_GENOME + LEN_BRAIN_GENOME]
    return body, brain_pool


def get_io_sizes_for_graph(graph) -> tuple[int, int, mj.MjModel, mj.MjData]:
    """
    Compile this body in OlympicArena and return:
      input_size  = len(qpos) + len(qvel)
      output_size = model.nu
    Also return (model, data) so caller can simulate immediately.
    """
    world = OlympicArena()
    core  = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=[-0.8, 0.0, 0.28])
    model = world.spec.compile()
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)
    return input_size, output_size, model, data


def pick_brain_slice_for_this_body(brain_pool: np.ndarray,
                                   input_size: int,
                                   output_size: int,
                                   *,
                                   hidden: int = HIDDEN_SIZE) -> np.ndarray:
    """
    Direct slice mapping: compute required #weights for THIS body and
    take that many weights from the front of the fixed brain_pool.
    """
    L1 = input_size * hidden
    L2 = hidden * hidden
    L3 = hidden * output_size
    need = L1 + L2 + L3

    if brain_pool.size < need:
        # pad with small noise to guarantee length (should rarely happen)
        pad = np.random.default_rng(0).normal(0, 0.05, size=need - brain_pool.size)
        return np.concatenate([brain_pool, pad])

    return brain_pool[:need].copy()
