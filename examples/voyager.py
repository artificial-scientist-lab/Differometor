import differometor as df
from differometor.setups import voyager
import jax.numpy as jnp
import matplotlib.pyplot as plt

# use a predefined Voyager setup with one noise detector and two signal detectors
S, _ = voyager()

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *metadata = df.run(S, [("f", "frequency")], frequencies)

powers = df.power_detector(carrier)

# calculate the signal power at the detector ports
signals = df.signal_detector(carrier, signal)

df.visualize_setup(S,
                    "examples/results/voyager/layout.html",
                    powers=powers, 
                    signals=signals, 
                    frequencies=frequencies,
                    port_to_index=metadata[-1])

detector_signals = signals[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
detector_signals = detector_signals[0] - detector_signals[1]

# calculate the sensitivity
sensitivity = noise / jnp.abs(detector_signals)

plt.figure()
plt.loglog(frequencies, sensitivity)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.grid()
plt.tight_layout()
plt.savefig("examples/results/voyager/sensitivities.png")
