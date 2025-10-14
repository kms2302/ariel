# examples/a3/baseline_new.py
# 
# Random-search baseline for assignment A3. It's aligned with our cmaes_new.py file.
#
# What does this script:
# - It uses the exact same genome layout as our CMA-ES setup:
#   * BODY: 3 probability vectors of length 64 (type / connection / rotation)
#   * BRAIN: a fixed-size pool of MLP weights (we slice only what the body needs)
# - Every generation, we just sample a new random population and evaluate them.
#   There is no selection, no mutation and no adaptation: this is pure random search.
# - For each robot we:
#   1) decode the body -> make a MuJoCo graph/model
#   2) slice & reshape brain weights -> build controller
#   3) simulate for a number of seconds
#   4) Scoring the robot the same way as in CMA-ES by taking the negative distance to the target (XY plane)
#    and subtract a small penalty if it didn’t reach the target.
# - We log the best-of-generation and best-overall.
# - We also save the single best baseline robot (body JSON + W1/W2/W3) so we can replay it.

# Standard library
import time
from typing import Any, Optional
from pathlib import Path
import random

# Import third-party
import numpy as np
import pandas as pd
import wandb
import mujoco as mj
import numpy.typing as npt

# Import ARIEL modules from CI Group
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
	HighProbabilityDecoder,
	save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
	construct_mjspec_from_graph,
)

# Group 4: our modules in this directory
from cmaes_new import (
	Individual,
	make_random_body_genome,
	make_random_brain_genome,
	make_random_genome,
	get_len_required,
)

# --- CONFIGURATION CONSTANTS ---
# Body genome constants (3 probability vectors, size=64 each)
GENOTYPE_SIZE = 64
LEN_BODY_GENOME = 3 * GENOTYPE_SIZE  # Which is 192

# Controller (brain) pool dimensions (we slice to actual I/O sizes per body)
I_MAX = 35  # maximum input_size
O_MAX = 30  # maximum output_size
HIDDEN_SIZE = 8
LEN_BRAIN_GENOME = I_MAX * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * O_MAX  # 584

# Final fixed genome length (body + brain)
LEN_TOTAL_GENOME = LEN_BODY_GENOME + LEN_BRAIN_GENOME  # 776
NUM_OF_MODULES = 30

# Hyperparameters
POP_SIZE = 32
GENERATIONS = 850
SECONDS_1 = 1
SECONDS_2 = 15
SECONDS_3 = 60
SECONDS_4 = 100
SECONDS_5 = 160
THRESHOLD_1 = -5.15  # -5.15: it moves!
THRESHOLD_2 = -4.45  # -4.4: it arrived at rugged!
THRESHOLD_3 = -4.15  # -4.15: it moves over rugged!
THRESHOLD_4 = -3.45  # -3.4: it arrived at inclined!

# Positions
TARGET_POSITION = [5, 0, 0.5]
SPAWN_POS = np.array([-0.8, 0, 0.1])
RUGGED_PENALTY = 0.3
INCLINED_PENALTY = 1.3
RUGGED_POS = SPAWN_POS + np.array([RUGGED_PENALTY, 0, 0])
INCLINED_POS = SPAWN_POS + np.array([INCLINED_PENALTY, 0, 0])

# We randomly pick one of these for each individual
# The penalty only applies if we didn’t exactly reach the target
SPAWN_DATA = {
	"pos": SPAWN_POS,
	"penalty": 0.0,
}
RUGGED_DATA = {
	"pos": RUGGED_POS,
	"penalty": RUGGED_PENALTY,
}
INCLINED_DATA = {
	"pos": INCLINED_POS,
	"penalty": INCLINED_PENALTY,
}
SPAWN_OPTIONS = [SPAWN_DATA, RUGGED_DATA, INCLINED_DATA]

# The list of spawn positions for the environment setup (as lists/arrays)
SPAWN_POSITIONS = [data["pos"].tolist() for data in SPAWN_OPTIONS] # needed for OlympicArena.spawn

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
	"Generations": GENERATIONS,
	"Population Size": POP_SIZE,
}

# Random generator setup for reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

# Data setup
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
# -----------------------------------------------------------------------------------

class BaselineSystem:
	"""
	Manages the overall baseline process, including
	WandB logging, environment setup, and tracking of the best individual.
	"""

	def __init__(self, seed: int = SEED, config: dict = CONFIG):
		self.seed = seed
		self.rng = np.random.default_rng(self.seed)
		self.config = config
		self.seconds = SECONDS_1
		self.best_f_overall = -np.inf
		self.best_individual = None  # Stores the best Genome object

		# Setup paths
		now = time.strftime("%Y%m%dT%H%M%S")
		self.run_name = f"{now}"
		self.save_path_graph = DATA / f"best_body_robot_graph_{self.run_name}.json"
		self.save_path_w1 = DATA / f"best_brain_w1_{self.run_name}.npy"
		self.save_path_w2 = DATA / f"best_brain_w2_{self.run_name}.npy"
		self.save_path_w3 = DATA / f"best_brain_w3_{self.run_name}.npy"
		print(f"Save path of graph: {self.save_path_graph}")

		# Initialize WandB
		self.run = wandb.init(
			entity=ENTITY,
			project=PROJECT,
			name=self.run_name,
			config=self.config,
		)

	def _update_simulation_duration(self) -> None:
		"""Dynamically updates the simulation duration based on overall best fitness."""
		if self.seconds == SECONDS_4 and self.best_f_overall > THRESHOLD_4:
			self.seconds = SECONDS_5
		elif self.seconds == SECONDS_3 and self.best_f_overall > THRESHOLD_3:
			self.seconds = SECONDS_4
		elif self.seconds == SECONDS_2 and self.best_f_overall > THRESHOLD_2:
			self.seconds = SECONDS_3
		elif self.seconds == SECONDS_1 and self.best_f_overall > THRESHOLD_1:
			self.seconds = SECONDS_2

	def _fitness_function(self, history: list[tuple[float, float, float]], penalty: float) -> float:
		"""
		Calculates fitness (negative Cartesian distance to target).
		
		If the target is not exactly reached, 
		we apply a small penalty depending on the spawn section.
		"""
		xt, yt, zt = TARGET_POSITION
		xc, yc, zc = history[-1]

		# Minimize the distance --> maximize the negative distance
		cartesian_distance = np.sqrt(
			(xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
		)
		
		# Apply the penalty (which is a reduction in fitness)
		# Note: Since fitness is negative distance (maximization), a positive penalty 
		# must be subtracted to make the final fitness score lower (worse).
		if -cartesian_distance < 0.0:  # only if finish is not reached
			final_fitness = -cartesian_distance - penalty
		
		return final_fitness

	def _log_generation(self, g: int, gen_fitness: list[float], best_f_in_gen: float) -> dict:
		"""Logs generation data to WandB and returns a dictionary for raw data storage."""
		log_data = {
			"gen": g,
			"seconds": self.seconds,
			"best_f_in_gen": best_f_in_gen,
			"best_f_overall": self.best_f_overall,
		}
		self.run.log(log_data, step=g)

		raw_data = log_data.copy()
		raw_data["gen_fitness"] = gen_fitness
		return raw_data

	def _save_best_individual(self, individual: 'Individual') -> None:
		"""Saves the components of the best individual found so far."""
		self.best_individual = individual
		save_graph_as_json(individual.robot_graph, self.save_path_graph)
		np.save(self.save_path_w1, individual.w1)
		np.save(self.save_path_w2, individual.w2)
		np.save(self.save_path_w3, individual.w3)

	def run_baseline(self) -> None:
		"""Main loop for the baseline."""
		generations_data = []

		for g in range(GENERATIONS):
			population_vectors = []

			for _ in range(POP_SIZE):
				candidate_vector = make_random_genome(self.rng)
				population_vectors.append(candidate_vector)
			
			losses = []
			gen_fitness = []
			population_individuals = []

			for candidate_vector in population_vectors:
				individual = Individual(candidate_vector)
				individual.decode()
				population_individuals.append(individual)

				# Execute the simulation
				mj.set_mjcb_control(lambda m, d: individual.controller.set_control(m, d))
				self._update_simulation_duration()
				simple_runner(individual.model, individual.data, duration=self.seconds)
				f = self._fitness_function(
					individual.tracker.history["xpos"][0],
					individual.spawn_penalty,
				)
				losses.append(-f)  # CMA-ES minimizes, so we negate fitness
				gen_fitness.append(f)

				# Check and save the overall best individual
				if f > self.best_f_overall:
					self.best_f_overall = f
					self._save_best_individual(individual)

			best_f_in_gen = max(gen_fitness)
			generations_data.append(self._log_generation(g, gen_fitness, best_f_in_gen))

		self._finalize_run(generations_data)

	def _finalize_run(self, generations_data: list[dict]) -> None:
		"""Cleans up and logs final artifacts."""
		out_dir = Path("wandb_artifacts")
		out_dir.mkdir(exist_ok=True)
		file_path = out_dir / f"{self.run_name}_raw.parquet"
		df = pd.DataFrame(generations_data)
		df.to_parquet(file_path, index=False)

		artifact = wandb.Artifact(
			name=f"{self.run_name}-raw-data",
			type="raw_data",
			metadata={"generations": GENERATIONS, "num_rows": len(df)}
		)
		artifact.add_file(str(file_path))
		self.run.log_artifact(artifact)
		self.run.finish()


if __name__ == "__main__":
	system = BaselineSystem()
	system.run_baseline()
