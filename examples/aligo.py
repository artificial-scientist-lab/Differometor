import json
import numpy as np
import differometor as df
from differometor.setups import aligo
import jax.numpy as jnp
from differometor.components import signal_detector, power_detector
import matplotlib.pyplot as plt
from differometor.plot import visualize_setup

# use a predefined aLIGO setup with one noise detector and one signal detector
S, _ = aligo()

with open("examples/data/aligo_test_data.json", "r") as f:
    aligo_test_data = json.load(f)

# set the frequency range
frequencies = jnp.array(aligo_test_data["frequencies"])

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *metadata = df.run(S, [("f", "frequency")], frequencies)

powers = power_detector(carrier)

# calculate the signal power at the detector ports
signals = signal_detector(carrier, signal)

visualize_setup(S, 
                  "examples/results/aligo/layout.html",
                  powers=powers, 
                  signals=signals, 
                  frequencies=frequencies,
                  port_to_index=metadata[-1])

detector_signals = signals[detector_ports[0]]

# calculate the sensitivity
sensitivities = noise / jnp.abs(detector_signals)

np.testing.assert_allclose(np.log(sensitivities), np.log(aligo_test_data["sensitivities"]), atol=1e-6, rtol=0)

plt.figure()
plt.loglog(frequencies, sensitivities)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.grid()
plt.tight_layout()
plt.savefig("examples/results/aligo/sensitivities.png")
