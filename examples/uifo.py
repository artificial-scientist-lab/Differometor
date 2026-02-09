import json
import numpy as np
import differometor as df
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differometor.setups import Setup
from differometor.setups import voyager
from differometor.plot import visualize_setup


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
signals = df.signal_detector(carrier, signal)
signals = signals[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
signals = signals[0] - signals[1]

# calculate the sensitivity
voyager_sensitivity = noise / jnp.abs(signals)

print("Voyager sensitivity calculation done!")


### Load pre-trained UIFO and calculate sensitivity ###
#-----------------------------------------------------#

print("Calculating pre-trained UIFO sensitivity...")

# Load the pre-trained UIFO model (no balanced homodyne detection, so only one detector)
with open(f"examples/data/uifo_{UIFO_TYPE}.json", "r") as f:
    uifo = json.load(f)    

S = Setup.from_data(uifo)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *metadata = df.run(S, [("f", "frequency")], frequencies)

powers = df.power_detector(carrier)

if len(detector_ports) == 1:
    # calculate the signal power at the detector port
    signals = df.signal_detector(carrier, signal)
    detector_signals = signals[detector_ports].squeeze()
else:
    # calculate the signal power at the detector ports (balanced homodyne detection scheme)
    signals = df.signal_detector(carrier, signal)
    detector_signals = signals[detector_ports]
    detector_signals = detector_signals[0] - detector_signals[1]

df.visualize_setup(S,
                    "examples/results/uifo/layout.html",
                    powers=powers, 
                    signals=signals, 
                    frequencies=frequencies,
                    port_to_index=metadata[-1])

# calculate the sensitivity
uifo_sensitivity = noise / jnp.abs(detector_signals)

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
plt.savefig("examples/results/uifo/sensitivities.png")
