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
