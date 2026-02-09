import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
from differometor.components import signal_detector
import matplotlib.pyplot as plt
import numpy as np
import jax
import optax
import json
import os


### Calculate the target sensitivity ###
#--------------------------------------#

# use a predefined Voyager setup with one noise detector and two signal detectors
S, component_property_pairs = voyager()

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = signal_detector(carrier, signal)
powers = powers[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
powers = powers[0] - powers[1]

# calculate the sensitivity
target_sensitivity = noise / jnp.abs(powers)
target_loss = jnp.sum(jnp.log10(target_sensitivity))


### Start from random parameters and optimize the sensitivity ###
#---------------------------------------------------------------#


# specify the ranges for the properties to be optimized
property_bounds = {
    "reflectivity": [0, 1],
    "tuning": [0, 90],
    "db": [0.01, 20],
    "angle": [-180, 180],
    "power": [0.01, 200],
    "mass": [0.01, 200],
    "length": [1, 4000],
    "phase": [-180, 180],
}

# select properties to be optimized
optimized_properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length", "phase"]
optimization_pairs = []
for pair in component_property_pairs:
    if pair[1] in optimized_properties:
        optimization_pairs.append(pair)

# build the setup once and then reuse it during the optimization
simulation_arrays, detector_ports, *_ = df.run_build_step(
    S,
    [("f", "frequency")],
    frequencies,
    optimization_pairs,
)

# calculate the bounds for the properties to be optimized
bounds = np.array([[
    property_bounds[pair[1]][0], 
    property_bounds[pair[1]][1]] for pair in optimization_pairs]).T

# start from random parameters
initial_guess = jnp.array(np.random.uniform(-10, 10, len(optimization_pairs)))


def objective_function(optimized_parameters):
    # map the parameters to between 0 and 1 and then to their respective bounds
    optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)
    simulation_arrays["optimized_parameters"] = optimized_parameters
    carrier, signal, noise = df.simulate(**simulation_arrays)
    powers = signal_detector(carrier, signal)
    powers = powers[detector_ports]
    powers = powers[0] - powers[1]
    sensitivity = noise / jnp.abs(powers)

    # loss relative to target loss => loss < 0 is better than voyager setup
    return jnp.sum(jnp.log10(sensitivity)) - target_loss


grad_fn = jax.jit(jax.value_and_grad(objective_function))
# warmup the function to compile it
_ = grad_fn(initial_guess)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.1)
)
optimizer_state = optimizer.init(initial_guess)

best_loss, best_params = 1e10, initial_guess
params, no_improve_count, losses = initial_guess, 0, []

for i in range(50000):
    loss, grads = grad_fn(params)

    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

    if loss < best_loss - 1e-4:
        best_loss, best_params, no_improve_count = loss, params, 0
        print(f"Iteration {i}: New best loss = {loss}")
    else:
        no_improve_count += 1

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(float(loss))

    # if the loss has not improved (< best_loss - 1e-4) over 1000 iterations, stop the optimization
    if no_improve_count > 1000:
        break

folder = "examples/results/voyager_optimization"
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


### Calculate the sensitivity of the best found setup ###
#-------------------------------------------------------#

update_setup(best_params, optimization_pairs, bounds, S)

carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)
powers = signal_detector(carrier, signal)
powers = powers[detector_ports]
powers = powers[0] - powers[1]
sensitivity = noise / jnp.abs(powers)
loss = jnp.sum(sensitivity)

plt.figure()
plt.plot(frequencies, sensitivity, label="Optimized Sensitivity")
plt.plot(frequencies, target_sensitivity, label="Target Sensitivity")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{folder}/sensitivities.png")
