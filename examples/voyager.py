import differometor as df
from differometor.setups import voyager
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt

# use a predefined Voyager setup with one noise detector and two signal detectors
S, _ = voyager()

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
sensitivity = noise / jnp.abs(powers)

plt.figure()
plt.loglog(frequencies, sensitivity)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.grid()
plt.tight_layout()
plt.savefig("examples/results/voyager.png")
