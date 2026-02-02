"""
Utilities for parameter bounding, setup mutation, simulation evaluation, and sensitivity/loss analysis.

This module contains:
- Bounding functions that map unconstrained parameters into physical bounds.
- Helper functions to write values into a simulation setup (node or edge properties).
- Power extraction utilities for detectors and optical components.
- Sensitivity calculators for different noise models.

Notes
-----
- The code uses JAX arrays (`jax.numpy` as `jnp`) and is written to be compatible with JAX tracing
  in most places. Be careful when converting JAX values to Python floats inside code paths that
  are intended to be JIT-compiled.
- Several functions assume a specific port ordering and packing for components, as described in
  the `calculate_powers` docstring.
"""

import jax.numpy as jnp

from typing import Any, Callable, List, Sequence, Tuple, Union

from differometor.components import power_detector, demodulate_signal_power


def sigmoid_bounding(parameters: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Map unconstrained parameters to physical bounds using a logistic sigmoid transform.

    The mapping is:
        u = sigmoid(p) = 1 / (1 + exp(-p))   -> u in (0, 1)
        x = u * (upper - lower) + lower

    Parameters
    ----------
    parameters
        Unconstrained real-valued parameters.
    bounds
        Bounds array with shape (2, ...). See `default_bounding`.

    Returns
    -------
    jnp.ndarray
        Parameters mapped into the provided bounds.
    """
    sigmoid_parameters = 1 / (1 + jnp.exp(-parameters))
    return sigmoid_parameters * (bounds[1] - bounds[0]) + bounds[0]


def inverse_sigmoid_bounding(parameters: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Map parameters in physical bounds back to unconstrained space using the inverse sigmoid.

    This is the inverse of `sigmoid_bounding`. It maps parameters from [lower, upper]
    back to the real line.

    The mapping is:
        u = (x - lower) / (upper - lower)   -> u in (0, 1)
        p = log(u / (1 - u))

    Parameters
    ----------
    parameters
        Parameters in physical bounds.
    bounds
        Bounds array with shape (2, ...). See `default_bounding`.
        
    Returns
    -------
    jnp.ndarray
        Unconstrained parameters mapped from the provided bounds.
    """
    u = (parameters - bounds[0]) / (bounds[1] - bounds[0])
    return jnp.log(u / (1 - u))


def set_value(
    node: str,
    property_name: str,
    value: Any,
    setup: Any,
) -> None:
    """
    Set a property value on a setup node or edge.

    The setup is expected to expose:
    - `setup.nodes[name]` for node data
    - `setup.edges["source_target"]` for edge data

    Edge names are inferred by the convention that an underscore indicates an edge key in the
    form "source_target". If the provided name has no underscore, it is treated as a node.

    Parameters
    ----------
    node
        Node name (for example "m1") or edge identifier string "source_target".
    property_name
        Name of the property to set inside the component "properties" mapping.
    value
        Value to store for the property.
    setup
        The setup object that holds nodes and edges with mutable property dictionaries.

    Returns
    -------
    None
        This function mutates `setup` in place.
    """
    # Node name: no underscore present.
    if "_" not in node:
        try:
            setup.nodes[node]["properties"][property_name] = value
        except KeyError:
            setup.nodes[node]["properties"] = {property_name: value}
        return

    # Edge name: underscore present, interpreted as "source_target".
    source, target = node.split("_")
    edge_key = source + "_" + target
    try:
        setup.edges[edge_key]["properties"][property_name] = value
    except KeyError:
        setup.edges[edge_key]["properties"] = {property_name: value}


def calculate_powers(
    carrier_solution: jnp.ndarray,
    detector_indices: jnp.ndarray,
    mirror_indices: jnp.ndarray,
    beamsplitter_indices: jnp.ndarray,
    isolator_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate powers at detectors and selected components and summarize by "hard" and "soft" sides.

    This function extracts port powers from a carrier solution and then aggregates component powers
    by selecting the maximum power per "side". The concept of "hard side" and "soft side" is:

    - Hard side: the side (per component) that contains the larger maximum port power.
    - Soft side: the opposite side's maximum port power.
    - Isolators are appended to the soft-side list directly (no side comparison is performed).

    Port packing assumptions
    ------------------------
    - Beamsplitter ports are packed in groups of 8 ports:
        left in, left out, top in, top out, right in, right out, bottom in, bottom out
      The function treats:
        - left+top as side A (ports 0..3)
        - right+bottom as side B (ports 4..7)
    - Mirror ports are packed in groups of 4 ports:
        left in, left out, right in, right out
      The function treats:
        - left as side A (ports 0..1)
        - right as side B (ports 2..3)

    Parameters
    ----------
    carrier_solution
        Carrier field solution for the setup, as produced by the simulator.
    detector_indices
        Indices of detector ports in the flattened power array.
    mirror_indices
        Indices of mirror ports in the flattened power array.
    beamsplitter_indices
        Indices of beamsplitter ports in the flattened power array.
    isolator_indices
        Indices of isolator ports in the flattened power array.

    Returns
    -------
    hard_side_powers
        Concatenated maximum powers on the dominant sides of beamsplitters and mirrors.
        Shape depends on the number of components and the power array trailing dimensions.
    soft_side_powers
        Concatenated maximum powers on the opposite sides of beamsplitters and mirrors,
        followed by isolator powers.
    detector_powers
        Powers at detector ports.
    """
    carrier_powers = power_detector(carrier_solution)

    beamsplitter_powers = carrier_powers[beamsplitter_indices]
    mirror_powers = carrier_powers[mirror_indices]
    isolator_powers = carrier_powers[isolator_indices]

    # Reshape into groups of ports per component.
    beamsplitters = beamsplitter_powers.reshape(-1, 8, *beamsplitter_powers.shape[1:])

    beamsplitter_left_top = beamsplitters[:, :4]
    beamsplitter_right_bottom = beamsplitters[:, 4:]

    max_power_left_top = jnp.max(beamsplitter_left_top, axis=1)
    max_power_right_bottom = jnp.max(beamsplitter_right_bottom, axis=1)

    # Select dominant side per beamsplitter, and capture the opposite side maximum as well.
    is_left_top_dominant = max_power_left_top > max_power_right_bottom
    max_beamsplitter_power = jnp.where(is_left_top_dominant, max_power_left_top, max_power_right_bottom)
    opposite_beamsplitter_power = jnp.where(is_left_top_dominant, max_power_right_bottom, max_power_left_top)

    mirrors = mirror_powers.reshape(-1, 4, *mirror_powers.shape[1:])
    mirrors_left = mirrors[:, :2]
    mirrors_right = mirrors[:, 2:]

    max_power_left = jnp.max(mirrors_left, axis=1)
    max_power_right = jnp.max(mirrors_right, axis=1)

    is_left_dominant = max_power_left > max_power_right
    max_mirrors_power = jnp.where(is_left_dominant, max_power_left, max_power_right)
    opposite_mirrors_power = jnp.where(is_left_dominant, max_power_right, max_power_left)

    hard_side_powers = jnp.concatenate([max_beamsplitter_power, max_mirrors_power], axis=0)
    soft_side_powers = jnp.concatenate(
        [opposite_beamsplitter_power, opposite_mirrors_power, isolator_powers], axis=0
    )
    detector_powers = carrier_powers[detector_indices]

    return hard_side_powers, soft_side_powers, detector_powers


def update_setup(
    parameters: jnp.ndarray,
    optimization_pairs: Sequence[Union[Tuple[str, str], List[Tuple[str, str]]]],
    bounds: jnp.ndarray,
    setup: Any,
    bounding_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = sigmoid_bounding,
) -> None:
    """
    Apply a parameter vector to a setup by writing bounded values to specified properties.

    Each entry in `optimization_pairs` specifies where to write the bounded value:
    - If the entry is a tuple (component_name, property_name), one property is updated.
    - If the entry is a list of tuples, the same bounded value is written to each listed target.

    Parameters
    ----------
    parameters
        One-dimensional array of raw optimization parameters. Length must match
        `len(optimization_pairs)`.
    optimization_pairs
        A sequence where each element is either:
        - (component_name, property_name), or
        - [(component_name, property_name), ...] for applying the same value to multiple targets.
    bounds
        Bounds with shape (2, number_of_parameters). `bounds[:, index]` is used for the
        corresponding parameter.
    setup
        The setup object to mutate in place.
    bounding_function
        Function that maps a raw parameter and its bounds to a physical value.

    Returns
    -------
    None
        This function mutates `setup` in place.
    """
    for index, optimization_pair in enumerate(optimization_pairs):
        # Convert to a Python float for storage in the setup object.
        value = float(bounding_function(parameters[index], bounds[:, index]))

        if isinstance(optimization_pair[0], list):
            # A list of (component_name, property_name) pairs.
            for component_name, property_name in optimization_pair:  # type: ignore[misc]
                set_value(component_name, property_name, value, setup)
        else:
            # A single (component_name, property_name) pair.
            component_name, property_name = optimization_pair  # type: ignore[misc]
            set_value(component_name, property_name, value, setup)


def calculate_sensitivities(
    results: Sequence[Tuple[Any, Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    sensitivity_function: Callable[[List[jnp.ndarray], List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
    frequencies: jnp.ndarray,
    homodyne: bool = True,
) -> jnp.ndarray:
    """
    Compute setup sensitivities from simulation results.

    Each element of `results` is expected to contain:
        (carrier_solution, signal_solution, noise, detector_indices, mirror_indices,
         beamsplitter_indices, isolator_indices)

    Signal extraction:
    - Signal powers are computed by demodulating the signal against the carrier solution.
    - Only detector ports are used for the sensitivity calculation.
    - For balanced homodyne detection, the signal is taken as the absolute difference between
      the first two detectors (assumed to be the homodyne pair).
    - For non-homodyne detection, the signal is taken from the first detector.

    Parameters
    ----------
    results
        Simulation results, one element per noise model or per setup variant as expected
        by `sensitivity_function`.
    sensitivity_function
        Callable that returns sensitivity given:
        - a list of noise arrays,
        - a list of signal power arrays,
        - and the frequencies array.
    frequencies
        Frequency grid used in the simulations. Shape is typically (number_of_ranges, number_of_points)
        but depends on the rest of your pipeline.
    homodyne
        If True, treat the first two detector channels as a balanced homodyne pair.

    Returns
    -------
    jnp.ndarray
        Sensitivity array as returned by `sensitivity_function`.
    """
    noises: List[jnp.ndarray] = []
    powers: List[jnp.ndarray] = []

    for result in results:
        carrier_solution, signal_solution, noise, detector_indices = result[0], result[1], result[2], result[3]

        # Demodulate using carrier and signal solutions.
        signal_powers = demodulate_signal_power(carrier_solution, signal_solution)

        # Extract signal powers at the detector ports.
        signal_powers = signal_powers[detector_indices]

        if homodyne:
            # Balanced homodyne: assume two detectors.
            signal_powers = jnp.abs(signal_powers[0] - signal_powers[1])
        else:
            # Single-ended detection: use the first detector.
            signal_powers = jnp.abs(signal_powers[0])

        powers.append(signal_powers)
        noises.append(noise)

    return sensitivity_function(noises, powers, frequencies)


def sensitivity_qamplfreq_noise(
    noises: List[jnp.ndarray],
    powers: List[jnp.ndarray],
    frequencies: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sensitivity model combining quantum, amplitude, and frequency noise.

    Expected inputs:
    - noises[0]: quantum noise
    - powers[0]: signal power used for normalization
    - powers[1]: signal power proxy for amplitude noise scaling
    - powers[2]: signal power proxy for frequency noise scaling

    The amplitude and frequency noise terms are constructed as:
    - amplitude_noise = powers[1] * 4e-9
    - frequency_noise = (powers[2] * frequencies) * 1e-8
      (implemented with transpose to match broadcasting expectations)

    Parameters
    ----------
    noises
        List of noise arrays.
    powers
        List of signal power arrays (see above).
    frequencies
        Frequency grid used to scale the frequency noise contribution.

    Returns
    -------
    jnp.ndarray
        Quadrature sum of noise contributions, normalized by the main signal power.
    """
    q_noise = jnp.abs(noises[0])
    amplitude_noise = powers[1] * 4e-9
    frequency_noise = (powers[2].T * frequencies).T * 1e-8
    return jnp.sqrt(q_noise**2 + amplitude_noise**2 + frequency_noise**2) / powers[0]

