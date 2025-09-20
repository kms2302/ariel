# exp1_ga.py
# =========================
# EXPERIMENT 1 (GA)
# Task: Making the gecko walk as far as possible in the environment BoxyRugged.
# Fitness = XY displacement of the "core" body after a fixed number of steps.
# =========================

import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from ariel.simulation.environments.boxy_heightmap import BoxyRugged
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Hyperparameters
TIMESTEPS   = 2000
GENERATIONS = 30
POP_SIZE    = 32
SEEDS       = [42, 1337, 2025]
MUT_RATE    = 0.1   # per-gene mutation rate
MUT_STDEV   = 0.1   # mutation step size

AMP_MAX = math.pi / 2


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="valid")


def make_model_and_data():
    mujoco.set_mjcb_control(None)
    world = BoxyRugged()
    body = gecko()
    world.spawn(body.spec, spawn_position=[0, 0, 0.08])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    core = [data.bind(g) for g in geoms if "core" in g.name][0]
    return model, data, core


def read_model_meta():
    model, data, _ = make_model_and_data()
    return model.nu, model.opt.timestep


NUM_JOINTS, DT = read_model_meta()


def unpack_params(theta):
    assert theta.size == NUM_JOINTS * 2 + 1
    A_raw   = theta[:NUM_JOINTS]
    phi_raw = theta[NUM_JOINTS:2*NUM_JOINTS]
    f_raw   = theta[-1]
    A = AMP_MAX * (1.0 / (1.0 + np.exp(-A_raw)))
    phi = (phi_raw % (2.0 * math.pi))
    f = 0.8 + 1.4 * (1.0 / (1.0 + np.exp(-f_raw)))
    return A, f, phi


def controller(theta, t):
    A, f, phi = unpack_params(theta)
    u = A * np.sin(2.0 * math.pi * f * t + phi)
    return np.clip(u, -math.pi/2, math.pi/2)


def rollout_fitness(theta):
    model, data, core = make_model_and_data()
    start_xy = core.xpos[:2].copy()
    t = 0.0
    for _ in range(TIMESTEPS):
        data.ctrl[:] = controller(theta, t)
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    end_xy = core.xpos[:2].copy()
    return float(np.linalg.norm(end_xy - start_xy))

def initialize_population(rng):
    A0   = rng.normal(0.0, 0.2, (POP_SIZE, NUM_JOINTS))
    phi0 = rng.uniform(0.0, 2.0*math.pi, (POP_SIZE, NUM_JOINTS))
    f0   = rng.normal(0.0, 0.3, (POP_SIZE, 1))
    return np.concatenate([A0, phi0, f0], axis=1)


def tournament_selection(pop, fitness, rng, k=3):
    """Pick one parent using k-way tournament."""
    idxs = rng.integers(0, len(pop), k)
    best_idx = idxs[np.argmax(fitness[idxs])]
    return pop[best_idx]


def crossover(p1, p2, rng):
    """One-point crossover."""
    point = rng.integers(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutate(child, rng):
    mask = rng.random(len(child)) < MUT_RATE
    child[mask] += rng.normal(0, MUT_STDEV, np.sum(mask))
    return child


def run_one_seed(seed):
    rng = np.random.default_rng(seed)
    pop = initialize_population(rng)
    fitness = np.array([rollout_fitness(ind) for ind in pop])
    best_per_gen = []

    for g in range(GENERATIONS):
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fitness, rng)
            p2 = tournament_selection(pop, fitness, rng)
            c1, c2 = crossover(p1, p2, rng)
            c1 = mutate(c1.copy(), rng)
            c2 = mutate(c2.copy(), rng)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:POP_SIZE])
        fitness = np.array([rollout_fitness(ind) for ind in pop])
        gen_best = np.max(fitness)
        best_per_gen.append(gen_best)
        print(f"[seed {seed}] gen {g+1:02d}/{GENERATIONS}  best={gen_best:.4f} m")

    return np.array(best_per_gen), float(np.max(fitness))


def main():
    curves, bests = [], []
    for seed in SEEDS:
        curve, best = run_one_seed(seed)
        curves.append(curve)
        bests.append(best)

    curves = np.vstack(curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)

    print("\nBest distances per seed:", [f"{b:.4f}" for b in bests])
    print(f"Final mean (gen {GENERATIONS}) = {mean[-1]:.4f} ± {std[-1]:.4f} m")

    xs = np.arange(1, GENERATIONS+1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, mean, label="GA (mean of 3 runs)")
    plt.fill_between(xs, mean-std, mean+std, alpha=0.25, label="±1 std")
    smoothed = moving_average(mean, w=5)
    xs_smooth = np.arange(1, len(smoothed)+1)
    plt.plot(xs_smooth, smoothed, linewidth=2, label="GA (mean, moving avg w=5)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness = XY displacement (m)")
    plt.title("Experiment 1 — Genetic Algorithm on Gecko (BoxyRugged)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("__results__/exp1_ga.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
