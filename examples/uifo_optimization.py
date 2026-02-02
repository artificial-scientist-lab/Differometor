import os
import jax
import json
import optax
import copy
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differometor.setups import voyager, uifo, constrain_inter_grid_cell_spaces
from differometor.utils import (
    sigmoid_bounding, 
    inverse_sigmoid_bounding,
    sensitivity_qamplfreq_noise, 
    calculate_sensitivities, 
    calculate_powers
)
from differometor.components import HARD_SIDE_POWER_THRESHOLD, SOFT_SIDE_POWER_THRESHOLD, DETECTOR_POWER_THRESHOLD
from differometor.simulate import run_setups, run_build_step, simulate


def calculate_loss(
        sensitivities, 
        reference_sensitivities, 
        powers
    ):
    # calculate power violations (i.e. penalty based on much the power at each component exceeds its threshold)
    hard_side_violations = jnp.maximum(powers[0] / HARD_SIDE_POWER_THRESHOLD - 1, 0).squeeze(1)
    soft_side_violations = jnp.maximum(powers[1] / SOFT_SIDE_POWER_THRESHOLD - 1, 0).squeeze(1)
    detector_violations = jnp.maximum(powers[2] / DETECTOR_POWER_THRESHOLD - 1, 0).squeeze(1)

    violations = jnp.concatenate([
        hard_side_violations,
        detector_violations,
        soft_side_violations],
        axis=0
    )

    losses = jnp.mean(jnp.log10(sensitivities.T / reference_sensitivities), axis=-1)
    penalties = jnp.sum(violations.T, axis=-1)

    return losses, penalties, violations


### Calculate the target sensitivity ###
#--------------------------------------#

print("Calculating target sensitivity...")

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 50)

# use a predefined Voyager setup with three different modulations (i.e. quantum noise, amplitude noise, frequency noise)
setups = [voyager(mode="space_modulation")[0], voyager(mode="amplitude_modulation")[0], voyager(mode="frequency_modulation")[0]]

# choose a sensitivity function that calculates sensitivities taking into account the three noise sources
sensitivity_function = sensitivity_qamplfreq_noise

# simulate the setups
simulation_results = run_setups(setups, frequencies)

# calculate the sensitivity values taking into account the three noise sources
reference_sensitivities = calculate_sensitivities(simulation_results, sensitivity_function, frequencies)

# calculate the light power at all components within the setup
powers = calculate_powers(simulation_results[0][0], *simulation_results[0][3:])

# calculate the loss taking into account power violations
sensitivity_loss, penalty, _ = calculate_loss(reference_sensitivities, reference_sensitivities, powers)
reference_loss = float(sensitivity_loss + penalty)

print("Target sensitivity calculation done!")


### Start from random parameters and optimize the sensitivity ###
#---------------------------------------------------------------#

# select properties to be optimized
optimized_properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]

# specify the ranges for the properties to be optimized
property_bounds = {
    "db": [0, 10],
    "angle": [-360, 360],
    "power": [0, 200],
    "tuning": [-360, 360],
    "mass": [0.01, 200],
    "length": [0.1, 4000],
    "reflectivity": [0, 1],
}

# random seed for reproducability
random_seed = 42

# define a random uifo with three different modulations (i.e. quantum noise, amplitude noise, frequency noise)
q_noise_setup, component_property_pairs, centers, boundaries = uifo(size=3, 
                                                                    mode="space_modulation", 
                                                                    random=True, 
                                                                    verbose=True, 
                                                                    random_seed=random_seed)
ampl_noise_setup, _ = uifo(size=3, mode="amplitude_modulation", centers=centers, boundaries=boundaries)
freq_noise_setup, _ = uifo(size=3, mode="frequency_modulation", centers=centers, boundaries=boundaries)

# make sure the base setup never accidentially changes
q_noise_setup_function = lambda: copy.deepcopy(q_noise_setup) 
ampl_noise_setup_function = lambda: copy.deepcopy(ampl_noise_setup)
freq_noise_setup_function = lambda: copy.deepcopy(freq_noise_setup)
setups = lambda: [q_noise_setup_function(), ampl_noise_setup_function(), freq_noise_setup_function()]

# couple vertical and horizontal spaces at same positions, so that the grid structure of the uifo is always preserved
optimization_pairs = constrain_inter_grid_cell_spaces(component_property_pairs, optimized_properties)

# calculate the bounds for the properties to be optimized
lower_bounds = []
upper_bounds = []
for optimization_pair in optimization_pairs:
    if isinstance(optimization_pair[0], list):
        property_name = optimization_pair[0][1]
    else:
        property_name = optimization_pair[1]
    lower_bounds.append(property_bounds[property_name][0])
    upper_bounds.append(property_bounds[property_name][1])
bounds = np.array([lower_bounds, upper_bounds])

print("\nRandomly initializing uifo parameters")
initial_guess = inverse_sigmoid_bounding(np.random.uniform(bounds[0], bounds[1], size=(len(bounds[0]),)), bounds)

# check if the random uifo uses a balanced homodyne detection scheme
homodyne = False
for node in q_noise_setup_function().nodes:
    if node[1]["component"] == "qhd":
        homodyne = True

# build the three modulation setups
print("\nBuilding...")
q_arrays, *q_metadata = run_build_step(q_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
ampl_arrays, *ampl_metadata = run_build_step(ampl_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
freq_arrays, *freq_metadata = run_build_step(freq_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
print("Building done!")


def objective_function(parameters):
    # map the parameters to between 0 and 1 and then to their respective bounds
    bounded_parameters = sigmoid_bounding(parameters, bounds)

    # simulate the three modulation setups
    for array in [q_arrays, ampl_arrays, freq_arrays]:
        array["optimized_parameters"] = bounded_parameters

    q_results = simulate(**q_arrays)
    ampl_results = simulate(**ampl_arrays)
    freq_results = simulate(**freq_arrays)
    results = [(*q_results, *q_metadata), (*ampl_results, *ampl_metadata), (*freq_results, *freq_metadata)]

    # calculate the sensitivities taking into account the three noise sources
    sensitivities = calculate_sensitivities(results, sensitivity_function, frequencies, homodyne=homodyne)

    # calculate the light power at all components within the setup
    powers = calculate_powers(q_results[0], *q_metadata)

    # calculate the loss taking into account power violations
    sensitivity_loss, penalty, _ = calculate_loss(sensitivities, reference_sensitivities, powers)
    penalty = penalty / (1.0 + penalty)
    return sensitivity_loss + penalty, (sensitivity_loss, penalty, sensitivities, powers)


grad_fn = jax.jit(jax.value_and_grad(objective_function, has_aux=True))
# warmup the function to compile it
print("\nCompiling...")
_ = grad_fn(initial_guess)
print("Compilation done!")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.1)
)
optimizer_state = optimizer.init(initial_guess)

best_loss, best_params = 1e10, initial_guess
params, losses = initial_guess, []

print("\nOptimizing... (only 100 iterations for demonstration)\n")
for i in range(100):
    start = time.time()
    (loss, (sensitivity_loss, penalty, sensitivities, powers)), grads = grad_fn(params)

    if loss < best_loss - 1e-4:
        best_loss, best_params = loss, params
        print(f"Iteration {i}: New best loss = {float(loss):.4f}, Penalty = {float(penalty):.4f}, Time = {(time.time()-start):.4f}s")
    else:
        print(f"Iteration {i}: Loss = {float(loss):.4f}, Penalty = {float(penalty):.4f}, Time = {(time.time()-start):.4f}s")

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(float(loss))

print("\nOptimization done!\n")
print("Evaluating...")

folder = "examples/results/uifo_optimization"
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)

with open(f"{folder}/parameters.json", "w") as f:
    json.dump(best_params.tolist(), f, indent=4)

with open(f"{folder}/losses.json", "w") as f:
    json.dump(losses, f, indent=4)

plt.figure()
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.axhline(0, color="red", linestyle="--")
plt.grid()
plt.tight_layout()
plt.savefig(f"{folder}/losses.png")
plt.close()

(loss, (sensitivity_loss, penalty, sensitivities, powers)), grads = grad_fn(best_params)

def plot_powers(powers, cutoff, name):
    powers = powers.squeeze()
    x = np.arange(powers.shape[0])

    plt.figure()
    plt.bar(x, powers)
    plt.axhline(y=cutoff, color="r", linestyle="--")
    plt.yscale("log")
    plt.xlabel("component")
    plt.ylabel("Power [W]")

    plt.tight_layout()
    plt.savefig(f"{folder}/powers_{name}.png")
    plt.close()

plot_powers(powers[0], HARD_SIDE_POWER_THRESHOLD, "hard_side")
plot_powers(powers[1], SOFT_SIDE_POWER_THRESHOLD, "soft_side")
plot_powers(powers[2], DETECTOR_POWER_THRESHOLD, "detector")


plt.figure()
plt.plot(frequencies, sensitivities, label="Optimized Sensitivity")
plt.plot(frequencies, reference_sensitivities, label="Target Sensitivity")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{folder}/sensitivities.png")
plt.close()


print(f"Evaluation done! You can find the results in the {folder} directory.")
print(f"Loss of best setup: {loss}")
