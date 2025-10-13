# Standard library
import time
from typing import Any, Optional
from pathlib import Path
import random

# Import third-party
import cma
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

# --- CONFIGURATION CONSTANTS ---
# Body genome constants (3 probability vectors, size=64 each)
GENOTYPE_SIZE = 64
LEN_BODY_GENOME = 3 * GENOTYPE_SIZE  # 192

# Brain sizing policy for a fixed-length CMA-ES genome
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
SIGMA_INIT = 0.7
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
SPAWN_POS = np.array([-0.8, 0, 0.1]) # Use numpy array for easy addition
RUGGED_PENALTY = 0.3
INCLINED_PENALTY = 1.3
RUGGED_POS = SPAWN_POS + np.array([RUGGED_PENALTY, 0, 0])
INCLINED_POS = SPAWN_POS + np.array([INCLINED_PENALTY, 0, 0])

# Map positions to penalties (for use in selection)
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
SPAWN_OPTIONS = [SPAWN_DATA, RUGGED_DATA, INCLINED_DATA] # List of dictionaries

# The list of spawn positions for the environment setup (as lists/arrays)
SPAWN_POSITIONS = [data["pos"].tolist() for data in SPAWN_OPTIONS] # Still needed for OlympicArena.spawn

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
	"Generations": GENERATIONS,
	"Population Size": POP_SIZE,
	"Initial Sigma": SIGMA_INIT,
}

# Random generator setup
SEED = 395
RNG = np.random.default_rng(SEED)

# Data setup
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
# -----------------------------------------------------------------------------------


class EvolutionarySystem:
	"""
	Manages the overall co-evolutionary process, including CMA-ES,
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
		"""Calculates fitness (negative Cartesian distance to target) and applies penalty."""
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

	def run_evolution(self) -> None:
		"""Main loop for the CMA-ES co-evolution."""
		x0 = make_random_genome(self.rng)  # Initial genome vector
		es = cma.CMAEvolutionStrategy(
			x0,
			SIGMA_INIT,
			{'popsize': POP_SIZE, 'seed': self.seed, 'verbose': -9},
		)

		generations_data = []

		for g in range(GENERATIONS):
			population_vectors = es.ask()
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

			es.tell(solutions=population_vectors, function_values=losses)

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


class Individual:
	"""
	Represents a single individual in the population with its full genome,
	decoded body (robot_graph, model, data, tracker), and decoded brain (controller).
	"""

	def __init__(self, genome: npt.NDArray[np.float64]):
		self.genome = genome
		# Decoded components (initialized to None)
		self.robot_graph = None
		self.model = None
		self.data = None
		self.tracker = None
		self.controller = None
		self.w1: Optional[npt.NDArray] = None
		self.w2: Optional[npt.NDArray] = None
		self.w3: Optional[npt.NDArray] = None
		self.spawn_data: dict = random.choice(SPAWN_OPTIONS) 
		self.spawn_position: list[float] = self.spawn_data["pos"].tolist()
		self.spawn_penalty: float = self.spawn_data["penalty"]

	def decode(self):
		"""Performs the 2-stage decoding of the full genome (body then brain)."""
		self._decode_body_genome()
		self._setup_simulation_env()
		self._decode_brain_genome()
		self._construct_controller()

	def _decode_body_genome(self):
		"""Decodes the body genome into a robot graph."""
		body_genome = self.genome[:LEN_BODY_GENOME]
		type_p_genes = body_genome[0:GENOTYPE_SIZE]
		conn_p_genes = body_genome[GENOTYPE_SIZE:2 * GENOTYPE_SIZE]
		rot_p_genes = body_genome[2 * GENOTYPE_SIZE:3 * GENOTYPE_SIZE]

		nde_input = [type_p_genes, conn_p_genes, rot_p_genes]

		nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
		p_matrices = nde.forward(nde_input)

		hpd = HighProbabilityDecoder(NUM_OF_MODULES)
		self.robot_graph = hpd.probability_matrices_to_graph(
			p_matrices[0], p_matrices[1], p_matrices[2]
		)

	def _setup_simulation_env(self):
		"""Sets up the MuJoCo model, data, and tracker based on the robot graph."""
		mj.set_mjcb_control(None)  # Important to avoid leftover controllers

		world = OlympicArena()
		core = construct_mjspec_from_graph(self.robot_graph)
		world.spawn(core.spec, position=self.spawn_position)

		self.model = world.spec.compile()
		self.data = mj.MjData(self.model)
		mj.mj_resetData(self.model, self.data)

		self.tracker = Tracker(
			mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core"
		)
		self.tracker.setup(world.spec, self.data)

	def _decode_brain_genome(self):
		"""Decodes the brain genome using direct slice mapping."""
		input_size = len(self.data.qpos)
		output_size = self.model.nu

		len_required, l1, l2, l3 = get_len_required(input_size, output_size)
		end_idx = LEN_BODY_GENOME + len_required
		brain_genome_slice = self.genome[LEN_BODY_GENOME:end_idx]

		# Extract weights and reshape them
		self.w1 = brain_genome_slice[:l1].reshape((input_size, HIDDEN_SIZE))
		self.w2 = brain_genome_slice[l1:l1 + l2].reshape((HIDDEN_SIZE, HIDDEN_SIZE))
		self.w3 = brain_genome_slice[l1 + l2:l1 + l2 + l3].reshape(
			(HIDDEN_SIZE, output_size)
		)

	def _construct_controller(self):
		"""Builds the Controller instance using the decoded weights."""
		w1, w2, w3 = self.w1, self.w2, self.w3
		
		# Check if weights are set (should always be true after _decode_brain_genome)
		if w1 is None or w2 is None or w3 is None:
			raise ValueError("Weights w1, w2, or w3 are not set before constructing the controller.")

		def nn_controller(
			model: mj.MjModel, data: mj.MjData
		) -> npt.NDArray[np.float64]:
			inputs = data.qpos
			layer1 = np.tanh(np.dot(inputs, w1))
			layer2 = np.tanh(np.dot(layer1, w2))
			outputs = np.tanh(np.dot(layer2, w3))
			return outputs * np.pi

		self.controller = Controller(
			controller_callback_function=nn_controller,
			tracker=self.tracker,
		)

# --- HELPER FUNCTIONS (that don't require class state) ---

def make_random_body_genome(rng: np.random.Generator) -> np.ndarray:
	"""A random body genome."""
	return rng.uniform(low=-100, high=100, size=LEN_BODY_GENOME)

def make_random_brain_genome(rng: np.random.Generator) -> np.ndarray:
	"""A random brain genome pool."""
	return rng.normal(loc=0.0138, scale=0.5, size=LEN_BRAIN_GENOME)

def make_random_genome(rng: np.random.Generator) -> np.ndarray:
	"""Make a random full genome for a single individual."""
	body_genome = make_random_body_genome(rng)
	brain_genome = make_random_brain_genome(rng)
	return np.concatenate([body_genome, brain_genome])

def get_len_required(input_size: int, output_size: int) -> tuple[int, int, int, int]:
	"""Calculates the required length and segment lengths for the brain genome."""
	l1 = input_size * HIDDEN_SIZE
	l2 = HIDDEN_SIZE * HIDDEN_SIZE
	l3 = HIDDEN_SIZE * output_size
	len_required = l1 + l2 + l3
	return len_required, l1, l2, l3

# ---------------------------------------------------------


if __name__ == "__main__":
	system = EvolutionarySystem()
	system.run_evolution()
