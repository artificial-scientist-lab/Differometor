# Differometor

[![PyPI version](https://img.shields.io/pypi/v/differometor)](https://pypi.org/project/differometor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Install guide**](#install-guide)
| [**Overview**](#overview)
| [**Examples**](#examples)
| [**Documentation**](#documentation)
| [**Development**](#development)
| [**Citing Differometor**](#citing-differometor)

<img src="media/differometor.png" alt="workflow" width="400"/>

### Differometor is a differentiable frequency domain interferometer simulator

You can use it to:

* Run high-speed interferometer optimizations with hundreds of parameters
* Simulate plane waves in quasi-static, user-specified interferometer configurations 
* Calculate light field modulations, quantum noise and optomechanical effects

Differometor is implemented in Python using the [JAX](https://docs.jax.dev/en/latest/index.html) framework and closely follows the design of the established [Finesse simulator](https://finesse.ifosim.org/).  

📄 **Documentation**: Read details about the inner workings of Differometor [here](media/documentation.pdf).

<img src="media/workflow.png" alt="workflow" width="500"/>


## Install Guide

The installation was tested on Ubuntu 24.04.1 LTS and on SUSE Linux Enterprise Server 15 SP6 with Python 3.11. The installation of Differometor will automatically install jax, optax, numpy and matplotlib. The installation was tested with jax==0.9.0, optax==0.2.6 and CUDA 13. Other package versions might work as well.

### From PyPI

This will install the CPU version of Differometor. The GPU version is significantly faster (see GPU support below). Create a virtual environment of your choice and install Differometor via pip. For example:

```bash
virtualenv venv
source venv/bin/activate
pip install differometor
```

### Development mode

This will install the CPU version of Differometor. The GPU version is significantly faster (see GPU support below). Git clone Differometor, create a virtual environment of your choice and install Differometor in edit mode. For example:

```bash
git clone https://github.com/artificial-scientist-lab/Differometor differometor
cd differometor
virtualenv venv
source venv/bin/activate
pip install -e .
```

### GPU support

Both installation methods above automatically install the CPU version of JAX. Differometor on GPU is significantly faster. To upgrade to the GPU version with CUDA 13, please activate your virtual environment and use:

```bash
pip install --upgrade "jax[cuda13]"
```


## Overview

Differometor’s implementation closely follows the design of the established [Finesse simulator](https://finesse.ifosim.org/). Differometor can simulate:

* Plane wave propagation in quasi-static, user-specified interferometer configurations
* Light field modulations / signal propagation
* Quantum noise
* Optomechanical effects

### Optimization Efficiency

The main difference to existing interferometer simulators is the speed of interferometer optimizations. Running auto-differentiated interferometer optimizations with Differometor on a Nvidia Quadro RTX 6000 GPU can be up to 160 times faster than running numerically differentiated optimizations with Finesse on a Xeon E5-2698 v4 CPU. 

<img src="media/speedup.png" alt="speedup" width="500"/>

### Accuracy

Differometor is verified against Finesse simulations, demonstrating close agreement in strain sensitivity curves of large interferometer setups. Below are some comparisons between strain sensitivity curves of a simplified aLIGO setup computed using Differometor and Finesse.

<img src="media/accuracy.png" alt="accuracy" width="1000"/>

### Differometor for the Computational Design of Gravitational Wave Detectors

Read the paper [Digital Discovery of Interferometric Gravitational Wave Detectors](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.021012) to understand how Differometor can be used to find impactful gravitational wave detector setups in quasi-universal interferometers (UIFO) like the one shown below.

<img src="media/uifo.png" alt="uifo" width="400"/>

## Examples

A good way to familiarize yourself with Differometor is to take a look at the [examples](examples):

   * [Optical Cavity Simulation](examples/cavity.py): Models a Fabry-Pérot cavity. Compare with the corresponding [Finesse example](https://finesse.ifosim.org/docs/latest/examples/01_simple_cav.html).
   * [Sensitivity of Advanced LIGO Setup](examples/aligo.py): Computes the strain sensitivity of a simplified aLIGO setup. Compare with the corresponding [Finesse example](https://finesse.ifosim.org/docs/latest/examples/09_aligo_sensitivity.html).
   * [Sensitivity of Voyager Setup](examples/voyager.py): Computes the strain sensitivity of the Voyager setup using balanced homodyne detection.
   * [Sensitivity Optimization of Voyager Setup](examples/voyager_optimization.py): Optimizes the sensitivity of the Voyager setup. 
   * [Sensitivity of a pre-optimized UIFO Setup](examples/uifo.py): Computes the strain sensitivity of a pre-optimized UIFO and compares it to Voyager.
   * [Sensitivity Optimization of UIFO Setup](examples/uifo_optimization.py): Optimizes the sensitivity of an UIFO.

### Optical Cavity Simulation 

This [example](examples/cavity.py) shows how Differometor can simulate a Fabry-Pérot cavity shown in the image below.

<img src="media/cavity_setup.png" alt="loss" width="400"/>

```python
import differometor as df
import jax.numpy as jnp
from differometor.components import power_detector
import matplotlib.pyplot as plt

# define a simple cavity setup with three detectors
S = df.Setup()
S.add("laser", "l0", power=1)
S.add("mirror", "m0", reflectivity=0.99, loss=0)
S.add("mirror", "m1", reflectivity=0.991, loss=0)
S.space("l0", "m0", length=1)
S.space("m0", "m1", length=1)
S.add("detector", "refl", target="m0", port="left", direction="out")
S.add("detector", "circ", target="m1", port="left", direction="in")
S.add("detector", "trns", target="m1", port="right", direction="out")

# set the tuning range
tunings = jnp.linspace(-180, 180, 400)

# run the simulation with the tuning as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("m0", "tuning")], tunings)

# calculate the power
powers = power_detector(carrier)

# plot the power at the detector ports
plt.figure()
plt.plot(tunings, powers[detector_ports[0]], label="refl")
plt.plot(tunings, powers[detector_ports[1]], label="circ")
plt.plot(tunings, powers[detector_ports[2]], label="trns")
plt.yscale("log")
plt.xlabel("Tuning (degrees)")
plt.ylabel("Power (W)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("examples/results/cavity.png")
```

It computes and plots the power outputs at the three detectors against the tuning of the left cavity mirror:

<img src="media/cavity_output.png" alt="loss" width="500"/>

### Sensitivity Optimization of Voyager Setup

This [example](examples/voyager_optimization.py) showcases the potential of Differometor when it comes to interferometer optimization. It uses the Voyager setup from above, initializes its parameters randomly and then optimizes the parameters to achieve the best possible sensitivity. On a Nvidia Quadro RTX 6000 GPU, this optimization takes around 2 minutes. Based on a few quick tests, around one third of the optimizations reach a loss < 0 which indicates that the sensitivity is better than in the Voyager design. This is without physical constraints, so such a solution probably burns the mirrors. 

```python
import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
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
powers = demodulate_signal_power(carrier, signal)
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
    powers = demodulate_signal_power(carrier, signal)
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
powers = demodulate_signal_power(carrier, signal)
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
```

The example output figures show that after 6000 iterations, the optimization successfully found a parameter set with which the Voyager setup has a better sensitivity than the original design. However, as we did not impose any physical constraints on the parameters, the optimized setup is likely not physically realizable. Differometor supports the implementation of such physical constraints (see [Sensitivity Optimization of UIFO Setup](examples/uifo_optimization.py)). 

<img src="media/loss.png" alt="loss" width="500"/>
<img src="media/sensitivity.png" alt="loss" width="500"/>


## Documentation

📄 **Documentation**: Read details about the inner workings of Differometor [here](media/documentation.pdf).

Differometor provides a Setup class to define the interferometer setup. You can add the following components:

* **mirror**
    * ```setup.add("mirror", "m0")``` adds a mirror component with name m0 and default properties. Names cannot contain underscores. 
    * ```setup.add("mirror", "m1", reflectivity=0.5)``` adds a mirror component with name m1 which reflects half of the incoming light and transmits the other half. All other properties are set to default.
    * ```setup.add("mirror", "abc", reflectivity=0.25, loss=0.5)``` adds a mirror component with name abc which reflects 25% of the incoming light, looses 50% of the incoming light and transmits the other 25%.
    * ```setup.add("mirror", "m3", transmissivity=0.3, loss=0.6, tuning=90)``` adds a mirror component with name m3 which transmits 30% of the incoming light, looses 60% of the incoming light and is [tuned](https://finesse.ifosim.org/docs/latest/physics/plane-waves/beam_splitter.html#tuning) by 90 degrees.
    * The default properties of a mirror are:
        * reflectivity: 0.5
        * loss: 5e-6
        * tuning: 0
    * A mirror has two ports: left and right. Each port has two directions: in and out.
    * Reflectivity, transmissivity and loss always add up to 1 and are given as fractions of the total light amplitude. Differometor expresses reflectivity and transmissivity as one reflectivity parameter which is the fraction of the remaining light after loss that is reflected. Reflectivity and transmissivity of mirror m3 in the example above would therefore be expressed as the fraction: (1-0.3-0.6)/(1-0.6) = 0.25 
* **beamsplitter**
    * A beamsplitter is handled equivalent to a mirror, but has an additional alpha parameter describing the angle of incidence of the incoming light and two more ports.
    * ```setup.add("beamsplitter", "bs", alpha=50)``` adds a beamsplitter component with name bs which reflects half of the incoming light and transmits the other half at an angle of 50 degrees. All other properties are set to default.
    * A beamsplitter has four ports: left, right, top and bottom. Each port has two directions: in and out. Incoming light on the left port is transmitted to the right port and reflected to the top port. Incoming light at the top is transmitted to the bottom port and reflected to the left port. Incoming light at the right port is transmitted to the left port and reflected to the bottom port. Incoming light at the bottom port is transmitted to the top port and reflected to the right port.

      <img src="media/beamsplitter.png" alt="beamsplitter" width="200"/>
    
    * The default properties of a beamsplitter are:
        * reflectivity: 0.5
        * loss: 5e-6
        * tuning: 0
        * alpha: 45
* **directional_beamsplitter**
    * ```setup.add("directional_beamsplitter", "dbs")``` adds a directional beamsplitter component with name dbs
    * A directional beamsplitter forwards all light at the left port to the right port, all light at the top port to the left port, all light at the right port to the bottom port and all light at the bottom port to the top port.
* **laser**
    * ```setup.add("laser", "l0")``` adds a laser component with name l0 and default properties.
    * ```setup.add("laser", "l1", power=0.5)``` adds a laser component with name l1 which has a power of 0.5. All other properties are set to default.
    * lasers can have a target if you want to connect them to a component without using a space. ```setup.add("laser", "l2", target="m0", port="left", direction="in")``` adds a laser component with name l2 which is directly connected to the left port of m0 in the in direction. All other properties are set to default.
    * The default properties of a laser are:
        * power: 1
        * phase: 0
* **squeezer**
    * ```setup.add("squeezer", "sq", db=10, angle=0)``` adds a squeezer with name sq producing a squeezed-light beam with given squeezing in decibels db and angle.
    * Like a laser above, a squeezer can also have an additional target, port and direction.
    * The default properties of a squeezer are:
        * db: 0
        * angle: 90
* **free_mass**
    * ```setup.add("free_mass", "sus", mass=10, target="m0")``` adds a simple free mass suspension with name sus and mass 10kg to the object with name m0. Masses can be added to mirrors or beamsplitters.
    * The default properties of a free mass are:
        * mass: 40
* **signal**
    * ```setup.add("signal", "s0", target="l0_amplitude")``` adds a amplitude signal generator with name s0 to the laser l0. 
    * ```setup.add("signal", "s1", target="abc_frequency")``` adds a frequency signal generator with name s1 to the laser abc.
    * ```setup.add("signal", "GW", target="m0_m1")``` adds a signal generator with name GW to the space between m0 and m1.
    * ```setup.add("signal", "s2", target="l0_amplitude", amplitude=("l0_power", jnp.sqrt))``` adds a signal generator with name s2 to the laser l0. The signal amplitude is defined as the square root of the laser power. So instead of actual values, properties can be defined as functions of other properties.
    * Signal generators can be connected only to spaces or lasers. For lasers, they can only be applied as amplitude (l0_amplitude) or frequency (l0_frequency) modulation. 
    * The default properties of a signal are:
        * amplitude: 1
        * phase: 0
* **qhd**
    * ```setup.add("qhd", "qhd0", detector1="noise1", detector2="noise2", phase=45)``` adds a balanced homodyne quantum noise detector with name qhd0 and a phase of 45 degrees. It uses the two noise detectors noise1 and noise2.
    * The default properties of a qhd are:
        * phase: 0
* **frequency**
    * ```setup.add("frequency", "f")``` sets the frequency used by all signals in the simulation to the default 1 Hz. It is a unique element and can only be set once in a given setup.
    * The default properties of a frequency are:
        * frequency: 1
* **detector**
    * ```setup.add("detector", "d0", target="bs", port="top", direction="out")``` adds a detector with name d0 to the top port of the object bs in the out direction.
    * ```setup.add("detector", "d1", target="m0")``` adds a detector with name d1 to the object m0. The default port and direction is left and in.
* **qnoised**
    * ```setup.add("qnoised", "1", target="m0")``` adds a noise detector with name 1 to the left port of the object m0 in the in direction. 
    * ```setup.add("qnoised", "noise", target="m0", port="left", direction="in", auxiliary=True)``` adds a noise detector with name noise to the left port of the object m0 in the in direction. This is an auxiliary detector that does not get its own noise output and can be used as detector1 or detector2 in a qhd.
* **nothing**
    * ```setup.add("nothing", "n0")``` adds a nothing component with name n0. These have two ports, left and right. Each port has two directions: in and out. Nothing components can e.g. be used to connect lasers and detectors without mirrors or beamsplitters in between.
    * The left input goes to the right output and the right input goes to the left output. Nothing changes.

You can connect components via spaces:

* **space**
    * ```setup.space("m0", "m1", length=0.5)``` adds a space between the objects m0 and m1 with a length of 0.5 meters. By default it connects the right port of m0 with the left port of m1.
    * ```setup.space("bs1", "bs2", length=100, source_port="top", target_port="bottom")``` adds a space between the objects bs1 and bs2 with a length of 100 meters. It connects the top port of bs1 with the bottom port of bs2.
    * ```setup.space("l0", "m0", length=1, refractive_index=1.5, target_port="right")``` adds a space between the laser l0 and the object m0 with a length of 1 meter and a refractive index of 1.5. It connects the laser l0 with the right port of m0, in the in direction.
    * The default properties of a space are:
        * length: 0
        * refractive_index: 1


## Development

Differometor is developed at the [Artificial Scientist Lab](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/research/research/cluster-research-groups/professorships/machine-learning-in-science-ii/) under Prof. Mario Krenn at the University of Tübingen, Germany.


## Citing Differometor

To cite this repository, please use the following BibTeX entry:

```
@software{differometor2025github,
  author = {Jonathan Klimesch and Yehonathan Drori and Rana X Adhikari and Mario Krenn},
  title = {Differometor: A Differentiable Interferometer Simulator for the Computational Design of Gravitational Wave Detectors},
  url = {http://github.com/artificial-scientist-lab/Differometor},
  version = {0.0.3},
  year = {2025},
}
```
