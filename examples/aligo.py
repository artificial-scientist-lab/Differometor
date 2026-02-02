import json
import numpy as np
import differometor as df
from differometor.setups import aligo
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt

# use a predefined aLIGO setup with one noise detector and one signal detector
S, _ = aligo()

with open("examples/data/aligo_test_data.json", "r") as f:
    aligo_test_data = json.load(f)

# set the frequency range
frequencies = jnp.array(aligo_test_data["frequencies"])

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports[0]]

# calculate the sensitivity
sensitivities = noise / jnp.abs(powers)

np.testing.assert_allclose(np.log(sensitivities), np.log(aligo_test_data["sensitivities"]), atol=1e-6, rtol=0)

plt.figure()
plt.loglog(frequencies, sensitivities)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.grid()
plt.tight_layout()
plt.savefig("examples/results/aligo.png")
