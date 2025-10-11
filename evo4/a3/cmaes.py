# Standard library
from typing import Any

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
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

"""
CO-EVOLUTION of a robot's body and its brain

1.  They are INTERDEPENDENT: A good controller for one body might be 
    terrible for another, and a particular body shape might only perform 
    well with a specific control strategy.

2.  Our GENOTYPE consists of TWO PARTS:
    A.  BODY GENOME
    B.  BRAIN GENOME

3.  The MAIN DIFFICULTY: the variable I/O size of the controller: different 
    random bodies of the same GENOTYPE_SIZE require different input_size 
    and output_size.

4.  The SOLUTION, ensure that:
    A.  the BRAIN GENOME only specifies the weights and structure of 
        the network.
    B.  the dimensions (input/output size) are determined by the body and 
        the simulation environment before the neural network is constructed.
"""

# --- DETERMINING THE TOTAL GENOME SIZE ---
# balancing robustness (making sure every possible body can have a brain) 
# and efficiency (not wasting too many genes).

# Body genome constants (3 probability vectors, size=64 each)
GENOTYPE_SIZE = 64
LEN_BODY_GENOME = 3 * GENOTYPE_SIZE  # 192

# Brain sizing policy for a fixed-length CMA-ES genome
I_MAX = 35  # maximum input_size
O_MAX = 30  # maximum output_size
HIDDEN_SIZE = 8
LEN_BRAIN_GENOME = I_MAX*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*O_MAX  # 584

# Final fixed genome length (body + brain)
LEN_TOTAL_GENOME = LEN_BODY_GENOME + LEN_BRAIN_GENOME  # 776
# -----------------------------------------

# Hyperparameters
POP_SIZE = 32
GENERATIONS = 32
SIGMA_INIT = 0.7
SECONDS = 15

# The Weights & Biases parameters
ENTITY = "evo4"
PROJECT = "assignment3"
CONFIG = {
    "Generations": GENERATIONS,
    "Population Size": POP_SIZE,
    "Initial Sigma": SIGMA_INIT,
}

# Random generator setup
SEED = 42
RNG = np.random.default_rng(SEED)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def make_random_body_genome() -> np.ndarray:
	"""
	A random body genome is a concatenated list of 3 vectors
	(each has size 64, so size 192 in total).
	"""
	return RNG.uniform(low=-100, high=100, size=LEN_BODY_GENOME)

def make_random_brain_genome() -> np.ndarray:
	"""
	A random brain genome is the pool from which controller weights are 
	extracted. The pool includes an unused tail that is ignored by design.
	"""
	return RNG.normal(loc=0.0138, scale=0.5, size=LEN_BRAIN_GENOME)

def make_random_genome() -> tuple[np.ndarray, np.ndarray]:
	"""
	Make a random genome for a single individual:
	- body: first 192 genes (= 3 * 64)
	- brain pool: remaining 584 genes
	"""
	body_genome = make_random_body_genome()
	brain_genome = make_random_brain_genome()
	return np.concatenate([body_genome, brain_genome])

def decode_genome(genome):
	"""
	The 2-STAGE DECODING process:
    A.  Decode BODY GENOME
    B.  Decode BRAIN GENOME using DIRECT SLICE MAPPING.

	Maps the extracted controller weights W into the matrices to 
	construct the Controller instance and returns it.
	"""
	robot_graph = decode_body_genome(genome)
	w1, w2, w3, model, data, tracker = decode_brain_genome(robot_graph, genome)

	def nn_controller(
		model: mj.MjModel,
		data: mj.MjData,
	) -> npt.NDArray[np.float64]:
		# Get inputs, in this case the positions of the actuator motors (hinges)
		inputs = data.qpos

		# Run the inputs through the lays of the network.
		layer1 = np.tanh(np.dot(inputs, w1))
		layer2 = np.tanh(np.dot(layer1, w2))
		outputs = np.tanh(np.dot(layer2, w3))

		# Scale the outputs
		return outputs * np.pi
	
	ctrl = Controller(
		controller_callback_function=nn_controller,
		tracker=tracker,
	)

	return ctrl, model, data, robot_graph, tracker

def decode_body_genome(genome: np.ndarray):
	"""
	Decoding the body genome from the full genome into the 
	robot graph that is used to decode the brain genome.
	"""
	# Extract the body genome from the full genome
	body_genome = genome[:LEN_BODY_GENOME]

	# Extract the genes for input to the NDE
	type_p_genes = body_genome[0:GENOTYPE_SIZE]
	conn_p_genes = body_genome[GENOTYPE_SIZE:2*GENOTYPE_SIZE]
	rot_p_genes = body_genome[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE]

	# Wrap the NDE input in a list
	nde_input = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]  # represents a single individual (= input for NDE)

	# Create the high-probability matrices
	nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
	p_matrices = nde.forward(nde_input)

    # Decode the high-probability graph
	hpd = HighProbabilityDecoder(NUM_OF_MODULES)
	robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

	return robot_graph

def get_io_sizes_for_graph(robot_graph) -> tuple[int, int]:
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

	# Reset state and time of simulation
	mj.mj_resetData(model, data)

	# Setting up a tracker to follow the "core" body part
	tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
	tracker.setup(world.spec, data)

	# Determine input and output size for the controller weights
	input_size = len(data.qpos)
	output_size = model.nu

	return input_size, output_size, model, data, tracker

def decode_brain_genome(robot_graph, genome):
	"""
	DECODING the BRAIN GENOME using DIRECT SLICE MAPPING. Since the required 
    number of weights varies, you can't simply take a fixed-length string 
    of genes and assign them directly
	
	UNUSED GENES: Since the total genome length must be fixed, 
    the genome x passed by CMA-ES will be longer than end_idx. 
    The genes beyond end_idx are unused genes for that individual.
	"""
	# Extract the brain genome from the full genome
	input_size, output_size, model, data, tracker = get_io_sizes_for_graph(robot_graph)
	len_required, l1, l2, l3 = get_len_required(input_size, output_size)
	end_idx = LEN_BODY_GENOME + len_required
	brain_genome = genome[LEN_BODY_GENOME:end_idx]

	w1 = brain_genome[:l1].reshape((input_size, HIDDEN_SIZE))
	w2 = brain_genome[l1:l1+l2].reshape((HIDDEN_SIZE, HIDDEN_SIZE))
	w3 = brain_genome[l1+l2:l1+l2+l3].reshape((HIDDEN_SIZE, output_size))

	return w1, w2, w3, model, data, tracker

def get_len_required(input_size: int, output_size: int) -> tuple[int, int, int, int]:
	l1 = input_size * HIDDEN_SIZE
	l2 = HIDDEN_SIZE * HIDDEN_SIZE
	l3 = HIDDEN_SIZE * output_size

	len_required = l1 + l2 + l3

	return len_required, l1, l2, l3


def main() -> None:
	# Start a new wandb run to track this script.
    run_name = f"CMA-ES-seed{SEED}"
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        config=CONFIG,
    )

    x0 = make_random_genome()

	# CMA for evolving this body (find best body for random brains)
    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA_INIT,
        {'popsize': POP_SIZE, 'seed': SEED, 'verbose': -9},  # quiet logs
    )
    generations = []
    best_per_gen = []
    best_f_overall = -np.inf  # best fitness seen overall

    for g in range(GENERATIONS):
        population = es.ask()
        losses = []
        gen_fitness = []

        for candidate in population:
            ctrl, model, data, robot_graph, tracker = decode_genome(candidate)

            # Telling MuJoCo to call our controller every timestep
            mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

            # Running the simulation for a short amount of time (fast test, no visualization)
            simple_runner(model, data, duration=SECONDS)
            f = fitness_function(tracker.history["xpos"][0])  # Displacement (bigger is better)
            losses.append(-f)  # Negate. Lower is better for CMA
            gen_fitness.append(f)

            if f > best_f_overall:
                best_f_overall = f
                best_body_graph = robot_graph

        es.tell(solutions=population, function_values=losses)

        # For plotting: keeping the best (i.e., max displacement) this generation
        best_f_in_gen = max(gen_fitness)
        best_per_gen.append(best_f_in_gen)

        # Log this gen (i.e., step) to Weights & Biases
        run.log({
            "gen": g,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        }, step=g)

        # Append raw rows for this generation
        generations.append({
            "gen": g,
            "gen_fitness": gen_fitness,
            "best_f_in_gen": best_f_in_gen, 
            "best_f_overall": best_f_overall,
        })

    if best_body_graph is not None:
        # Save the graph as JSON for later inspection
        save_graph_as_json(best_body_graph, DATA / f"best_body_{run_name}.json")
    
    # End of run: create a DataFrame and write to Parquet (or CSV)
    out_dir = Path("wandb_artifacts")
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / f"{run_name}_raw.parquet"
    df = pd.DataFrame(generations)
    df.to_parquet(file_path, index=False)

    # Create an artifact, add the file, and log it
    artifact = wandb.Artifact(
        name=f"{run_name}-raw-data",
        type="raw_data",
        metadata={"generations": GENERATIONS, "num_rows": len(df)}
    )
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)

    # Finish the run and upload any remaining data.
    run.finish()


if __name__ == "__main__":
    main()
