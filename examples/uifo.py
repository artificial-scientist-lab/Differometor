import json
import numpy as np
import differometor as df
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differometor.setups import Setup
from differometor.setups import voyager
from differometor.components import demodulate_signal_power


UIFO_TYPE = "800_3000" # options: "20_5000", "800_3000"

if UIFO_TYPE == "800_3000":
    with open(f"examples/data/uifo_{UIFO_TYPE}_test_data.json", "r") as f:
        uifo_test_data = json.load(f)


### Calculate the Voyager sensitivity ###
#---------------------------------------#

print("Calculating Voyager sensitivity...")

# use a predefined Voyager setup with one noise detector and two signal detectors
S, _ = voyager()

# set the frequency range
frequencies = jnp.array(uifo_test_data["frequencies"]) if UIFO_TYPE == "800_3000" else jnp.logspace(jnp.log10(float(UIFO_TYPE.split("_")[0])), jnp.log10(float(UIFO_TYPE.split("_")[1])), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
powers = powers[0] - powers[1]

# calculate the sensitivity
voyager_sensitivity = noise / jnp.abs(powers)

print("Voyager sensitivity calculation done!")


### Load pre-trained UIFO and calculate sensitivity ###
#-----------------------------------------------------#

print("Calculating pre-trained UIFO sensitivity...")

# Load the pre-trained UIFO model (no balanced homodyne detection, so only one detector)
with open(f"examples/data/uifo_{UIFO_TYPE}.json", "r") as f:
    uifo = json.load(f)    

S = Setup.from_data(uifo)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

if len(detector_ports) == 1:
    # calculate the signal power at the detector port
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports].squeeze()
else:
    # calculate the signal power at the detector ports (balanced homodyne detection scheme)
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports]
    powers = powers[0] - powers[1]

# calculate the sensitivity
uifo_sensitivity = noise / jnp.abs(powers)

if UIFO_TYPE == "800_3000":
    np.testing.assert_allclose(np.log(uifo_sensitivity), np.log(uifo_test_data["sensitivities"]), atol=1e-5, rtol=0)

print("Pre-trained UIFO sensitivity calculation done!")


### Compare both sensitivity curves ###
#-------------------------------------#

plt.loglog(frequencies, voyager_sensitivity, label="Voyager")
plt.loglog(frequencies, uifo_sensitivity, label="UIFO")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("examples/results/uifo.png")
