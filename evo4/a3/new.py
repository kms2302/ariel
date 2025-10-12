"""
CO-EVOLUTION of a robot's body and its brain

1.  They are INTERDEPENDENT: A good controller for one body might be 
    terrible for another, and a particular body shape might only perform 
    well with a specific control strategy.

2.  Our GENOTYPE consists of TWO PARTS:
    A.  BODY GENOME
    B.  BRAIN GENOME

3.  A random BODY is a list of 3 vectors (each has size GENOTYPE_SIZE=64)

4.  The MAIN DIFFICULTY: the variable I/O size of the controller: different 
    random bodies of the same GENOTYPE_SIZE require different input_size 
    and output_size.

5.  The SOLUTION, ensure that:
    A.  the BRAIN GENOME only specifies the weights and structure of 
        the network.
    B.  the dimensions (input/output size) are determined by the body and 
        the simulation environment before the neural network is constructed.

6.  The 2-STAGE DECODING process:
    A.  Decode BODY GENOME
    B.  Decode BRAIN GENOME using DIRECT SLICE MAPPING.

7.  CMA-ES requires a FIXED GENOME SIZE N:
        N = LEN_BODY_GENOME + LEN_BRAIN_GENOME

8.  DECODING the BODY GENOME from genome x:
    A.  EXTRACT the body genome: x[0:LEN_BODY_GENOME)]
    B.  DECODE the body genome into core and graph
    C.  DEFINE the body in Mujoco (model, data)
    D.  QUERY the model to retrieve the I/O size
            - input_size: len(data.qpos)
            - output_size: model.nu
    E.  CALCULATE LEN_REQUIRED: the total number of weights needed for the 
        network based on input_size and output_size.
            - L1 = input_size * hidden_size
            - L2 = hidden_size * hidden_size
            - L3 = hidden_size * output_size
            - total LEN_REQUIRED = L1 + L2 + L3

9.  DECODING the BRAIN GENOME using DIRECT SLICE MAPPING. Since the required 
    number of weights varies, you can't simply take a fixed-length string 
    of genes and assign them directly:
    A.  END_IDX = LEN_BODY GENOME + LEN_REQUIRED
    B.  EXTRACT the CONTROLLER WEIGHTS: x[LEN_BODY_GENOME:END_IDX]
    C.  UNUSED GENES: Since the total genome length, N, must be fixed, 
        the genome x passed by CMA-ES will be longer than END_IDX. 
        The genes beyond END_IDX are unused genes for that individual.
    D.  CONSTRUCT the CONTROLLER: Map the extracted controller weights W into
        the Input*Hidden and Hidden*Output matrices.
            - Map W1:
                - Initialize idx pointer: idx = 0
                - Flat W1 = W[idx:idx+L1]
                - W1 = Flat W1.reshape((input_size, hidden_size))
                - Update idx pointer: idx=idx+L1
            - Map W2:
                - Flat W2 = W[idx:idx+L2]
                - W2 = Flat W2.reshape((hidden_size, hidden_size))
                - Update idx pointer: idx=idx+L2
            - Map W3:
                - Flat W3 = W[idx:idx+L3]
                - W3 = Flat W3.reshape((hidden_size, output_size))

10. DETERMINING the total genome size N: balancing robustness (making sure 
    every possible body can have a brain) and efficiency (not wasting too 
    many genes).
    A.  N = LEN_BODY_GENOME + LEN_BRAIN_GENOME = 192 + 584 = 776
    B.  LEN_BODY_GENOME = 3 * GENOTYPE_SIZE = 3 * 64 = 192
    C.  LEN_BRAIN_GENOME = (I_MAX*8)+(8*8)+(8*O_MAX) = 584
        - I_MAX (maximum input_size): 35
        - O_MAX (maximum output_size): 30
"""
"""
new.py  —  Co-evolution representation (GENOME) + DECODING

What this file does (plain English):
- We define ONE flat genome vector that CMA-ES can optimize. It has two parts:
    [ BODY_GENOME | BRAIN_GENOME ]
- BODY_GENOME (first 3*GENOTYPE_SIZE numbers) are probabilities that get decoded
  into a robot body using our existing NDE -> HPD pipeline.
- BRAIN_GENOME (the rest) is a big bucket of weights for a tiny MLP controller.
  Since different bodies have different I/O sizes, we **slice** only as many
  weights as we need for the current body, and ignore the rest.
- The point is: CMA-ES wants a fixed dimension; we give it that, but only use the
  piece that fits the current body. Unused brain genes are simply ignored.

This matches Kevyn's plan (steps 5–9).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import mujoco as mj

# ---- ARIEL bits: environment + body construction from a graph ----
from ariel.simulation.environments import OlympicArena
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# ---- Our body generator/decoder (NDE -> HPD) ----
# We try to import constants+types from your team's body_evo; if something isn't
# there (names differ), we fall back to safe defaults so this file is self-contained.
try:
    from examples.a3.body_evo import (
        GENOTYPE_SIZE as BE_GENOTYPE_SIZE,
        NUM_OF_MODULES as BE_NUM_MODULES,
        Genotype as BodyGenotype,
        decode_to_graph,
    )
    _HAS_BODY_EVO = True
except Exception:
    # Fallbacks (these match your earlier code if imports change)
    BE_GENOTYPE_SIZE = 64
    BE_NUM_MODULES = 30
    BodyGenotype = None       # not used in fallback mode
    decode_to_graph = None    # will assert if used without being present
    _HAS_BODY_EVO = False

# ---- Controller helper: MLP from explicit weight matrices ----
# If the helper already exists in controller_auto.py we import it.
# If not, we define a local fallback with the exact same behavior.
try:
    from examples.a3.controller_auto import make_mlp_controller_from_weights
    _HAS_CTRL_HELPER = True
except Exception:
    _HAS_CTRL_HELPER = False

    def make_mlp_controller_from_weights(W1: np.ndarray, W2: np.ndarray, W3: np.ndarray):
        """Minimal fallback: wrap a 2-hidden-layer tanh MLP as an ARIEL Controller."""
        from ariel.simulation.controllers.controller import Controller  # local import
        in_dim, h = W1.shape
        assert W2.shape == (h, h)
        assert W3.shape[0] == h
        out_dim = W3.shape[1]

        def _cb(model: mj.MjModel, data: mj.MjData, **_):
            obs = np.concatenate([data.qpos, data.qvel])
            # be robust if input size doesn't match (shouldn't happen, but just in case)
            if obs.size != in_dim:
                if obs.size > in_dim:
                    obs = obs[:in_dim]
                else:
                    obs = np.pad(obs, (0, in_dim - obs.size))
            h1 = np.tanh(obs @ W1)
            h2 = np.tanh(h1 @ W2)
            u  = np.tanh(h2 @ W3) * (np.pi / 2)  # scale to ~±90°
            return u

        return Controller(controller_callback_function=_cb, tracker=None)

# =========================
# 1) Genome specification
# =========================

# Body part length = 3 * GENOTYPE_SIZE (type, connection, rotation)
LEN_BODY_GENOME = 3 * BE_GENOTYPE_SIZE

# Brain part length must be FIXED (upper bound across all bodies). We use:
#   input  <= I_MAX, output <= O_MAX, hidden = HIDDEN
# resulting brain length = (I_MAX*HIDDEN) + (HIDDEN*HIDDEN) + (HIDDEN*O_MAX)
I_MAX   = 35
O_MAX   = 30
HIDDEN  = 8
LEN_BRAIN_GENOME = (I_MAX * HIDDEN) + (HIDDEN * HIDDEN) + (HIDDEN * O_MAX)

# Final genome dimensionality seen by CMA-ES:
GENOME_DIM = LEN_BODY_GENOME + LEN_BRAIN_GENOME


@dataclass
class Genome:
    """Flat, fixed-length vector CMA-ES will mutate. Shape = (GENOME_DIM,)."""
    x: np.ndarray


def random_genome(rng: np.random.Generator) -> Genome:
    """
    Make a random genome:
      - body genes are probabilities in [0,1]
      - brain genes are small Gaussian weights
    """
    body = rng.random(LEN_BODY_GENOME)
    brain = rng.normal(0.0, 0.3, size=LEN_BRAIN_GENOME)
    return Genome(np.concatenate([body, brain]).astype(np.float64))


# =========================
# 2) Decoding – BODY first
# =========================

def _unpack_body_vectors(body_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the first 3*GENOTYPE_SIZE numbers into the 3 probability vectors expected
    by our NDE: type_vec, conn_vec, rot_vec (all clipped to [0,1]).
    """
    g = np.asarray(body_flat, dtype=np.float64).ravel()
    assert g.size == LEN_BODY_GENOME, "body_flat length mismatch"

    t = np.clip(g[0:BE_GENOTYPE_SIZE], 0.0, 1.0).astype(np.float32)
    c = np.clip(g[BE_GENOTYPE_SIZE:2*BE_GENOTYPE_SIZE], 0.0, 1.0).astype(np.float32)
    r = np.clip(g[2*BE_GENOTYPE_SIZE:3*BE_GENOTYPE_SIZE], 0.0, 1.0).astype(np.float32)
    return t, c, r


def decode_body_to_graph(body_flat: np.ndarray):
    """
    Use our team's NDE->HPD pipeline to turn (t,c,r) probability vectors into a
    concrete robot body **graph** (node-link DiGraph-like).
    """
    if decode_to_graph is None:
        raise RuntimeError("decode_to_graph not available. Import from body_evo first.")
    t, c, r = _unpack_body_vectors(body_flat)

    # body_evo.Genotype expects three vectors; we create the dataclass if present,
    # else we pass a simple object with attributes (works with your earlier code).
    if BodyGenotype is not None:
        geno = BodyGenotype(type_vec=t, conn_vec=c, rot_vec=r)
    else:
        geno = type("Genotype", (), {"type_vec": t, "conn_vec": c, "rot_vec": r})

    graph = decode_to_graph(geno)
    return graph


def build_model_from_graph(graph: Any, spawn_z: float = 0.30) -> Tuple[mj.MjModel, mj.MjData]:
    """
    Compile a MuJoCo model for the given body graph inside OlympicArena.
    Return (model, data). We keep this tiny and opinionated on purpose.
    """
    world = OlympicArena()
    core  = construct_mjspec_from_graph(graph)
    world.spawn(core.spec, position=[-0.8, 0.0, float(spawn_z)])
    model = world.spec.compile()
    data  = mj.MjData(model)
    return model, data


# ======================================
# 3) Decoding – BRAIN (slice + reshape)
# ======================================

def slice_brain_weights(
    brain_flat: np.ndarray,
    input_size: int,
    output_size: int,
    hidden_size: int = HIDDEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map the (fixed, possibly long) brain vector into exactly the 3 matrices we
    need for THIS body:
        W1: [input_size,  hidden]
        W2: [hidden,      hidden]
        W3: [hidden,      output_size]
    Extra values at the end of brain_flat are UNUSED; if brain_flat is too short
    (shouldn't happen with our bounds), we pad with zeros.
    """
    brain = np.asarray(brain_flat, dtype=np.float64).ravel()

    L1 = input_size * hidden_size
    L2 = hidden_size * hidden_size
    L3 = hidden_size * output_size
    need = L1 + L2 + L3

    if brain.size < need:
        brain = np.pad(brain, (0, need - brain.size))
    # else: we just ignore extra genes (CMA-ES still gets a fixed dimension)

    idx = 0
    W1 = brain[idx:idx+L1].reshape((input_size, hidden_size)); idx += L1
    W2 = brain[idx:idx+L2].reshape((hidden_size, hidden_size)); idx += L2
    W3 = brain[idx:idx+L3].reshape((hidden_size, output_size))
    return W1, W2, W3


# ================================================
# 4) Full decode: genome -> graph/model/controller
# ================================================

def decode_individual(genome: Genome, spawn_z: float = 0.30) -> dict[str, Any]:
    """
    End-to-end decoding used by CMA-ES:
      1) Split genome into body vs brain.
      2) Decode BODY -> graph -> (model,data)
      3) Read true I/O sizes from (model,data)
      4) Slice BRAIN weights and build an MLP controller for this body

    Returns a dict so downstream code can pick what it needs.
    """
    x = np.asarray(genome.x, dtype=np.float64).ravel()
    assert x.size == GENOME_DIM, f"Genome length must be {GENOME_DIM}, got {x.size}"

    body_flat  = x[:LEN_BODY_GENOME]
    brain_flat = x[LEN_BODY_GENOME:]

    # (1) body -> graph
    graph = decode_body_to_graph(body_flat)

    # (2) compile once to learn sizes
    model, data = build_model_from_graph(graph, spawn_z=spawn_z)

    input_size  = int(data.qpos.size + data.qvel.size)
    output_size = int(model.nu)

    # Defensive clamp: we promised our upper bounds cover all bodies
    assert input_size  <= I_MAX, f"input_size={input_size} exceeds I_MAX={I_MAX}"
    assert output_size <= O_MAX, f"output_size={output_size} exceeds O_MAX={O_MAX}"

    # (3) brain -> (W1,W2,W3) -> controller
    W1, W2, W3 = slice_brain_weights(brain_flat, input_size, output_size, hidden_size=HIDDEN)
    ctrl = make_mlp_controller_from_weights(W1, W2, W3)

    return {
        "graph": graph,
        "model": model,
        "data": data,
        "input_size": input_size,
        "output_size": output_size,
        "weights": (W1, W2, W3),
        "controller": ctrl,
    }


# =========================
# 5) Tiny smoke test (run)
# =========================

if __name__ == "__main__":
    """
    Run me like:
      PYTHONPATH=. uv run python -m examples.a3.new
    Expected: prints input/output sizes and steps the sim for ~0.5 s.
    """
    rng = np.random.default_rng(123)
    g = random_genome(rng)
    out = decode_individual(g)

    model, data, ctrl = out["model"], out["data"], out["controller"]
    print(f"I/O sizes: {out['input_size']} → {out['output_size']} | hidden={HIDDEN}")

    # quick warm-up (no control) then half-second with our MLP controller
    def _finite(d: mj.MjData) -> bool:
        return (
            np.isfinite(d.qpos).all() and
            np.isfinite(d.qvel).all() and
            np.isfinite(d.qacc).all() and
            float(np.max(np.abs(d.qacc))) < 2e4
        )

    # warm-up
    old = mj.get_mjcb_control()
    mj.set_mjcb_control(None)
    for _ in range(300):  # ~0.3 s at 1ms
        mj.mj_step(model, data)
        if not _finite(data): break

    # run with controller
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    for _ in range(300):  # ~0.3 s
        mj.mj_step(model, data)
        if not _finite(data): break
    mj.set_mjcb_control(old)

    print("Smoke test finished.")
