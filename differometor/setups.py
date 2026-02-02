"""
Interferometer setup builder and converter utilities.

This module defines:

- A small graph-like container (Setup) with:
  - nodes (optical and mechanical components)
  - edges (spaces connecting components)
  - a parameter list that can be used for optimization workflows

- Several predefined setups:
  - voyager
  - aligo
  - uifo (quasi-universal interferometer grid)

- Utilities:
  - constrain_inter_grid_cell_spaces (parameter tying for UIFO grids)
  - differometor_to_finesse (export to Finesse text format)

Design notes
------------
- Node names are not allowed to contain underscores. Underscores are reserved for
  edge identifiers of the form "source_target".
- Edges are stored internally with tuple keys (source, target) and are presented
  externally as strings "source_target" for convenience in some APIs.
- The "parameters" list stores (component_identifier, property_name) pairs.
  - For nodes, component_identifier is the node name.
  - For edges (spaces), component_identifier is "source_target".

- The `not_optimizable` argument can be:
  - None: treat as empty list (everything is optimizable by default)
  - True: mark the entire component as not optimizable
  - list: property-level exclusions. Each item must be a (name, property_name) pair.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from differometor.components import DEFAULT_PROPERTIES


# Public types used throughout this module
NodeName = str
EdgeKey = Tuple[NodeName, NodeName]          # internal representation
EdgeName = str                               # external "source_target" representation
PropertyName = str

# (component_identifier, property_name)
ParameterPair = Tuple[str, str]

NotOptimizable = Union[None, bool, List[ParameterPair]]


class Nodes:
    """
    Lightweight view wrapper over the internal node dictionary.

    Instances of this class are created and maintained by Setup. It provides:
    - iteration in a style similar to networkx: `for name, data in setup.nodes(data=True): ...`
    - `__getitem__` for direct lookup of a node data dictionary

    Parameters
    ----------
    nodes:
        The underlying node dictionary owned by Setup. Keys are node names and
        values are node metadata dictionaries.
    """

    def __init__(self, nodes: Dict[NodeName, Dict[str, Any]]):
        self._nodes = nodes

    def __iter__(self) -> Iterator[Tuple[NodeName, Dict[str, Any]]]:
        """Iterate over (node_name, node_data)."""
        return iter(self._nodes.items())

    def __call__(self, data: bool = True) -> Iterator[Any]:
        """
        Mimic a subset of the networkx `.nodes(data=...)` behavior.

        Parameters
        ----------
        data:
            If True, yield (node_name, node_data).
            If False, yield node_name.

        Returns
        -------
        iterator:
            Iterator over nodes according to `data`.
        """
        if data:
            return iter(self._nodes.items())
        return iter(self._nodes.keys())

    def __getitem__(self, node: NodeName) -> Dict[str, Any]:
        """
        Retrieve metadata for a given node.

        Raises
        ------
        KeyError:
            If `node` is not present.
        """
        if node in self._nodes:
            return self._nodes[node]
        raise KeyError(f"Node '{node}' not found in the setup.")


class Edges:
    """
    Lightweight view wrapper over the internal edge dictionary.

    Internally, edges are stored as keys `(source, target)`.

    Externally, some interfaces address edges by the string "source_target".
    This class supports both:
    - iteration similar to networkx: `for src, tgt, data in setup.edges(data=True): ...`
    - `__getitem__` with the string edge name "source_target"

    Parameters
    ----------
    edges:
        The underlying edge dictionary owned by Setup. Keys are (source, target)
        tuples and values are edge metadata dictionaries.
    """

    def __init__(self, edges: Dict[EdgeKey, Dict[str, Any]]):
        self._edges = edges

    def __iter__(self) -> Iterator[Tuple[NodeName, NodeName, Dict[str, Any]]]:
        """Iterate over (source, target, edge_data)."""
        return iter((src, tgt, data) for (src, tgt), data in self._edges.items())

    def __getitem__(self, edge: EdgeName) -> Dict[str, Any]:
        """
        Retrieve metadata for a given edge addressed as "source_target".

        Parameters
        ----------
        edge:
            String edge name in the format "source_target".

        Raises
        ------
        KeyError:
            If the edge is not present.
        ValueError:
            If `edge` does not contain a single underscore separator.
        """
        if "_" not in edge:
            raise ValueError(
                f"Edge '{edge}' must be addressed as 'source_target' (one underscore separator)."
            )
        source, target = edge.split("_", 1)
        if (source, target) in self._edges:
            return self._edges[(source, target)]
        raise KeyError(f"Edge '{edge}' not found in the setup.")

    def __call__(self, data: bool = True) -> Iterator[Any]:
        """
        Mimic a subset of the networkx `.edges(data=...)` behavior.

        Parameters
        ----------
        data:
            If True, yield (source, target, edge_data).
            If False, yield (source, target).

        Returns
        -------
        iterator:
            Iterator over edges according to `data`.
        """
        if data:
            return iter((src, tgt, d) for (src, tgt), d in self._edges.items())
        return iter((src, tgt) for (src, tgt) in self._edges)


class Setup:
    """
    Container describing an optical (and optomechanical) interferometer-like setup.

    The setup consists of:
    - nodes: components such as laser, mirror, beamsplitter, squeezer, detector, free_mass, signal
    - edges: spaces connecting nodes with length and refractive index properties
    - parameters: a flat list of (component_identifier, property_name) pairs that represent
      what can be optimized (unless excluded)

    Attributes
    ----------
    parameters:
        List of (component_identifier, property_name) pairs to optimize.
        For nodes, component_identifier is the node name.
        For spaces, component_identifier is the string "source_target".
    nodes:
        A Nodes view over the internal node dictionary.
    edges:
        An Edges view over the internal edge dictionary.
    default_properties:
        DEFAULT_PROPERTIES imported from differometor.components
    """

    def __init__(self):
        self.parameters: List[ParameterPair] = []
        self._nodes: Dict[NodeName, Dict[str, Any]] = {}
        self._edges: Dict[EdgeKey, Dict[str, Any]] = {}

        # Lightweight views that provide convenient iteration and lookup
        self.nodes = Nodes(self._nodes)
        self.edges = Edges(self._edges)

        # Property templates per component type
        self.default_properties = DEFAULT_PROPERTIES

    def add(
        self,
        component: str,
        name: str,
        not_optimizable: NotOptimizable = None,
        target: Optional[str] = None,
        port: Optional[str] = None,
        direction: Optional[str] = None,
        auxiliary: Optional[bool] = None,
        detector1: Optional[str] = None,
        detector2: Optional[str] = None,
        **properties: Any,
    ) -> None:
        """
        Add a node (component) to the setup.

        Parameters
        ----------
        component:
            Component type name. Must exist in DEFAULT_PROPERTIES.
            Common values include: "laser", "mirror", "beamsplitter", "squeezer",
            "frequency", "signal", "free_mass", "detector", "qnoised", "qhd",
            "directional_beamsplitter", "nothing".
        name:
            Node name identifier. Must not contain underscores. Underscores are reserved
            for edge identifiers "source_target".
        not_optimizable:
            Controls how this node contributes to `self.parameters`.

            Accepted values:
            - None: treat as an empty list, meaning all properties are optimizable
              (subject to special rules for certain components).
            - True: exclude this entire node from the parameter list.
            - list of (node_name, property_name): exclude individual properties.

            Special rule:
            - For "signal" and "frequency" components, parameters are not added by default.
        target:
            Optional target reference, used by certain components such as "free_mass",
            "signal", "detector", and "qnoised". Accepted formats:
            - node name, for example "bs"
            - edge name "source_target", for example "l0_prm"
            - node property "node_amplitude" or "node_frequency" for signal components
        port:
            Optional port name. Must be one of: "left", "top", "right", "bottom".
        direction:
            Optional direction string. Must be one of: "in", "out".
        auxiliary:
            Optional auxiliary flag. Must be True or False if provided.
        detector1, detector2:
            Optional detector node names used for "qhd" components.

        **properties:
            Component-specific properties, merged with DEFAULT_PROPERTIES[component].

            For "mirror" and "beamsplitter", an additional convenience is supported:
            - You may pass transmissivity instead of reflectivity.
              In that case, reflectivity is derived and normalized with respect to loss.

        Raises
        ------
        ValueError:
            For invalid component names, invalid property sets, invalid ports or directions,
            invalid targets, or underscore usage in node names.
        """
        if "_" in name:
            raise ValueError(
                f"Node name '{name}' cannot contain underscores. Use '-' instead."
            )

        # Normalize not_optimizable
        if not_optimizable is None:
            not_optimizable = []

        # Convenience: allow transmissivity for mirror and beamsplitter
        if component in ["mirror", "beamsplitter"]:
            if "reflectivity" in properties and "transmissivity" in properties:
                raise ValueError(
                    "Cannot specify both 'reflectivity' and 'transmissivity'. "
                    "Use 'reflectivity' and 'loss' or 'transmissivity' and 'loss' instead."
                )

            if "transmissivity" in properties:
                # Merge defaults first so loss is known
                properties = {**self.default_properties[component], **properties}
                transmissivity = properties.pop("transmissivity")

                # Convert transmissivity to normalized reflectivity with respect to loss.
                # Internally, reflectivity is stored as a factor on (1 - loss).
                properties["reflectivity"] = (1 - transmissivity - properties["loss"]) / (
                    1 - properties["loss"]
                )
            elif "reflectivity" in properties:
                # Merge defaults first so loss is known
                properties = {**self.default_properties[component], **properties}

                # Normalize reflectivity by (1 - loss) to store it in the internal convention.
                properties["reflectivity"] = properties["reflectivity"] / (
                    1 - properties["loss"]
                )

        # Merge defaults for every component
        try:
            properties = {**self.default_properties[component], **properties}
        except KeyError as exc:
            raise ValueError(f"Component '{component}' is not recognized.") from exc

        # Validate that no unknown properties were provided
        if len(properties) != len(self.default_properties[component]):
            raise ValueError(
                f"Component '{component}' has the properties "
                f"{list(self.default_properties[component].keys())} "
                f"but received {list(properties.keys())}."
            )

        # Add optimizable parameters for regular components
        if component not in ["signal", "frequency"]:
            if not_optimizable is not True:
                self.parameters.extend(
                    [
                        (name, property_name)
                        for property_name in properties.keys()
                        if (name, property_name) not in not_optimizable  # type: ignore[operator]
                    ]
                )

        # Store node
        self._nodes[name] = {"component": component, "properties": properties}

        # Parse and validate target references (if any)
        if target is not None:
            # Target might be edge-like "src_tgt" or property-like "node_amplitude"
            if "_" in target:
                suffix = target.split("_")[-1]

                # Edge targets: accept "src_tgt" where suffix is not "amplitude" or "frequency"
                if suffix not in ["amplitude", "frequency"]:
                    try:
                        self.edges[target]  # validate edge exists
                    except KeyError as exc:
                        raise ValueError(f"Target '{target}' is not in the setup.") from exc
                    self._nodes[name]["target"] = target

                # Property targets: "node_amplitude" or "node_frequency"
                else:
                    node_name = target.split("_", 1)[0]
                    try:
                        self.nodes[node_name]  # validate node exists
                    except KeyError as exc:
                        raise ValueError(f"Target '{node_name}' is not in the setup.") from exc
                    self._nodes[name]["target"] = node_name
                    self._nodes[name]["target_property"] = suffix
            else:
                # Plain node target
                try:
                    self.nodes[target]  # validate node exists
                except KeyError as exc:
                    raise ValueError(f"Target '{target}' is not in the setup.") from exc
                self._nodes[name]["target"] = target

        # Validate and store port and direction, when relevant
        if port is not None:
            if port not in ["left", "top", "right", "bottom"]:
                raise ValueError(
                    f"Port '{port}' is not recognized. Use 'left', 'top', 'right', or 'bottom'."
                )
            self._nodes[name]["port"] = port

        if direction is not None:
            if direction not in ["in", "out"]:
                raise ValueError(
                    f"Direction '{direction}' is not recognized. Use 'in' or 'out'."
                )
            self._nodes[name]["direction"] = direction

        if auxiliary is not None:
            if auxiliary not in [True, False]:
                raise ValueError(
                    f"Auxiliary '{auxiliary}' is not recognized. Use True or False."
                )
            self._nodes[name]["auxiliary"] = auxiliary

        # Validate detector references for components that depend on them
        if detector1 is not None:
            if detector1 not in self._nodes:
                raise ValueError(f"Detector1 '{detector1}' is not in the setup.")
            self._nodes[name]["detector1"] = detector1

        if detector2 is not None:
            if detector2 not in self._nodes:
                raise ValueError(f"Detector2 '{detector2}' is not in the setup.")
            self._nodes[name]["detector2"] = detector2

    def space(
        self,
        source: str,
        target: str,
        not_optimizable: NotOptimizable = None,
        source_port: str = "right",
        target_port: str = "left",
        **properties: Any,
    ) -> None:
        """
        Add a space (edge) between two existing nodes.

        Parameters
        ----------
        source:
            Source node name. Must exist in the setup.
        target:
            Target node name. Must exist in the setup.
        not_optimizable:
            Controls how this space contributes to `self.parameters`.

            Accepted values:
            - None: treat as empty list knowing all space properties are optimizable
            - True: exclude this entire space from parameter list
            - list of (edge_name, property_name) pairs to exclude individually

            For spaces, the component identifier in the parameter list is "source_target".
        source_port:
            Which port on the source node the space starts at. Default is "right".
        target_port:
            Which port on the target node the space ends at. Default is "left".
        **properties:
            Space properties, merged with DEFAULT_PROPERTIES["space"].
            The default properties usually include length and refractive_index.

        Raises
        ------
        ValueError:
            If source or target nodes do not exist, or if properties are invalid.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' is not in the setup.")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' is not in the setup.")

        if not_optimizable is None:
            not_optimizable = []

        # Merge and validate properties
        properties = {**self.default_properties["space"], **(properties or {})}
        if len(properties) != len(self.default_properties["space"]):
            raise ValueError(
                f"Space has the properties {list(self.default_properties['space'].keys())} "
                f"but received {list(properties.keys())}."
            )

        # Add optimizable parameters for spaces, unless excluded
        edge_name = f"{source}_{target}"
        if not_optimizable is not True:
            self.parameters.extend(
                [
                    (edge_name, property_name)
                    for property_name in properties.keys()
                    if (edge_name, property_name) not in not_optimizable  # type: ignore[operator]
                ]
            )

        # Store edge
        self._edges[(source, target)] = {
            "properties": properties,
            "source_port": source_port,
            "target_port": target_port,
        }

    def to_data(self) -> dict:
        """
        Serialize this setup into a plain Python dictionary.

        This is intended for:
        - saving to JSON (after ensuring all values are JSON-serializable)
        - transferring setups between processes

        Returns
        -------
        data:
            Dictionary with keys:
            - "parameters": list
            - "nodes": dict
            - "edges": dict with string keys "source_target"
        """
        edges_serializable = {
            f"{src}_{tgt}": data for (src, tgt), data in self._edges.items()
        }
        return {
            "parameters": list(self.parameters),
            "nodes": dict(self._nodes),
            "edges": edges_serializable,
        }

    @classmethod
    def from_data(cls, data: dict) -> "Setup":
        """
        Rebuild a Setup instance from the dictionary created by `to_data`.

        Parameters
        ----------
        data:
            Dictionary containing "parameters", "nodes", and "edges".

        Returns
        -------
        setup:
            A reconstructed Setup instance with Nodes and Edges view wrappers attached.

        Notes
        -----
        - Edge keys are expected as strings "source_target".
        - Node names may not contain underscores, so a split on the first underscore is safe.
        """
        setup = cls()
        setup.parameters = list(data["parameters"])
        setup._nodes = dict(data["nodes"])
        setup._edges = {
            tuple(edge_str.split("_", 1)): edge_data
            for edge_str, edge_data in data["edges"].items()
        }

        # Re-link view wrappers after replacing the underlying dictionaries
        setup.nodes = Nodes(setup._nodes)
        setup.edges = Edges(setup._edges)
        return setup


# ----------------------------------------------------------------------
# Predefined setups
# ----------------------------------------------------------------------

def voyager(mode: str = "space_modulation") -> tuple[Setup, list]:
    """
    Construct and return the predefined Voyager setup.

    Parameters
    ----------
    mode:
        Modulation type. Supported values:
        - "space_modulation"
        - "amplitude_modulation"
        - "frequency_modulation"

    Returns
    -------
    setup:
        The constructed setup instance.
    parameters:
        The list of optimizable parameters derived from the setup.

    Raises
    ------
    ValueError:
        If `mode` is not one of the supported values.
    """
    S = Setup()
    S.add("laser", "l0", power=153, phase=0)
    S.add("mirror", "prm", transmissivity=0.049, loss=5e-06, tuning=0)
    S.add("beamsplitter", "bs", transmissivity=0.5, loss=5e-06, tuning=63.63961030678928, alpha=45)
    S.add("mirror", "itmy", transmissivity=0.002, loss=5e-06, tuning=0)
    S.add("mirror", "etmy", transmissivity=1.5e-05, loss=5e-06, tuning=0)
    S.add("mirror", "itmx", transmissivity=0.002, loss=5e-06, tuning=0)
    S.add("mirror", "etmx", transmissivity=1.5e-05, loss=5e-06, tuning=0)
    S.add("mirror", "srm", transmissivity=0.046, loss=5e-06, tuning=90)
    S.add("directional_beamsplitter", "dbs1")
    S.add("directional_beamsplitter", "dbs2")
    S.add("squeezer", "sq", db=10, angle=0)
    S.add("mirror", "fm1", transmissivity=0.1e-2, loss=5e-06, tuning=0)
    S.add("mirror", "fm2", transmissivity=1.5e-05, loss=5e-06, tuning=-0.014)
    S.add("beamsplitter", "bhbs", transmissivity=0.5, loss=5e-06, tuning=1e-07, alpha=45)
    S.add("laser", "lo", power=0.01, phase=0)

    # Mechanical degrees of freedom (free-mass suspensions) targeting optical nodes
    S.add("free_mass", "prmsus", mass=29.243802983873618, target="prm")
    S.add("free_mass", "bssus", mass=48.634040943805395, target="bs")
    S.add("free_mass", "itmysus", mass=200, target="itmy")
    S.add("free_mass", "etmysus", mass=200, target="etmy")
    S.add("free_mass", "itmxsus", mass=200, target="itmx")
    S.add("free_mass", "etmxsus", mass=200, target="etmx")
    S.add("free_mass", "srmsus", mass=50, target="srm")

    # Optical connections (spaces)
    S.space("l0", "prm", length=1)
    S.space("prm", "bs", length=1)
    S.space("bs", "itmy", length=1, source_port="top")
    S.space("itmy", "etmy", length=4000)
    S.space("bs", "itmx", length=1, source_port="right")
    S.space("itmx", "etmx", length=4000)
    S.space("bs", "srm", length=10, source_port="bottom")
    S.space("srm", "dbs1", length=1, target_port="left")
    S.space("sq", "dbs2", length=1, target_port="top")
    S.space("dbs1", "dbs2", length=10, source_port="top", target_port="right")
    S.space("dbs2", "fm1", length=1, source_port="left")
    S.space("fm1", "fm2", length=300)
    S.space("dbs1", "bhbs", length=1, source_port="right", target_port="left")
    S.space("lo", "bhbs", length=10, target_port="bottom")

    # Frequency node used by signal generation in some workflows
    S.add("frequency", "f", frequency=1)

    # Signal definitions depend on the chosen modulation mode
    if mode == "space_modulation":
        # Signals target specific spaces (edges) by name "source_target"
        S.add("signal", "fl0prm", target="l0_prm")
        S.add("signal", "fprmbs", target="prm_bs")
        S.add("signal", "fbsitmy", target="bs_itmy", phase=180)
        S.add("signal", "fitmyetmy", target="itmy_etmy", phase=180)
        S.add("signal", "fbsitmx", target="bs_itmx")
        S.add("signal", "fitmxetmx", target="itmx_etmx")
        S.add("signal", "bssrm", target="bs_srm", phase=180)
    elif mode == "amplitude_modulation":
        # Signals target node properties by "node_amplitude"
        S.add("signal", "fl0", target="l0_amplitude", amplitude=("l0_power", jnp.sqrt))
        S.add("signal", "flo", target="lo_amplitude", amplitude=("lo_power", jnp.sqrt))
    elif mode == "frequency_modulation":
        # Signals target node properties by "node_frequency"
        S.add("signal", "fl0", target="l0_frequency")
        S.add("signal", "flo", target="lo_frequency")
    else:
        raise ValueError(
            "Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'."
        )

    # Balanced homodyne readout noise and detectors
    S.add("qnoised", "noise-top", target="bhbs", port="top", direction="out", auxiliary=True)
    S.add("qnoised", "noise-right", target="bhbs", port="right", direction="out", auxiliary=True)
    S.add("qhd", "noise", detector1="noise-top", detector2="noise-right", phase=180)
    S.add("detector", "detector-top", target="bhbs", port="top", direction="out")
    S.add("detector", "detector-right", target="bhbs", port="right", direction="out")

    return S, S.parameters


def aligo(mode: str = "space_modulation") -> tuple[Setup, list]:
    """
    Construct and return a simplified aLIGO setup with optomechanics and squeezing.

    Parameters
    ----------
    mode:
        Modulation type. Supported values:
        - "space_modulation"
        - "amplitude_modulation"
        - "frequency_modulation"

    Returns
    -------
    setup:
        The constructed setup instance.
    parameters:
        The list of optimizable parameters derived from the setup.

    Raises
    ------
    ValueError:
        If `mode` is not one of the supported values.
    """
    Larm = 3995
    itmT = 0.014
    mirrorL = 37.5e-6
    etmT = 5e-6
    Mtm = 40

    S = Setup()
    S.add("laser", "L0", power=125)
    S.add("beamsplitter", "bs", reflectivity=0.5, loss=0, alpha=45)
    S.add("mirror", "prm", transmissivity=0.03, loss=mirrorL, tuning=90)
    S.add("mirror", "itmx", transmissivity=itmT, loss=mirrorL, tuning=90)
    S.add("mirror", "etmx", transmissivity=etmT, loss=mirrorL, tuning=89.999875)
    S.add("mirror", "itmy", transmissivity=itmT, loss=mirrorL, tuning=0)
    S.add("mirror", "etmy", transmissivity=etmT, loss=mirrorL, tuning=0.000125)
    S.add("mirror", "srm", transmissivity=0.2, loss=mirrorL, tuning=-90)
    S.add("squeezer", "sq1", db=10, angle=90)

    # Mechanical degrees of freedom for the test masses
    S.add("free_mass", "itmxsus", mass=Mtm, target="itmx")
    S.add("free_mass", "etmxsus", mass=Mtm, target="etmx")
    S.add("free_mass", "itmysus", mass=Mtm, target="itmy")
    S.add("free_mass", "etmysus", mass=Mtm, target="etmy")

    # Optical connections
    S.space("L0", "prm")
    S.space("prm", "bs", length=53)
    S.space("bs", "itmx", length=4.5)
    S.space("itmx", "etmx", length=Larm)
    S.space("bs", "itmy", length=4.45, source_port="top")
    S.space("itmy", "etmy", length=Larm)
    S.space("bs", "srm", length=50.525, source_port="bottom")
    S.space("sq1", "srm", target_port="right")

    # Signal frequency
    S.add("frequency", "f", frequency=5)

    # Mode-dependent signal injection
    if mode == "space_modulation":
        S.add("signal", "darmx", target="itmx_etmx")
        S.add("signal", "darmy", target="itmy_etmy", phase=180)
    elif mode == "frequency_modulation":
        S.add("signal", "fL0", target="L0_frequency")
    elif mode == "amplitude_modulation":
        S.add("signal", "fL0", target="L0_amplitude", amplitude=("L0_power", jnp.sqrt))
    else:
        raise ValueError(
            "Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'."
        )

    # Readout noise and detector at the signal recycling mirror output port
    S.add("qnoised", "noise", target="srm", port="right", direction="out")
    S.add("detector", "detector", target="srm", port="right", direction="out")

    return S, S.parameters


def uifo(
    size: int,
    centers: Optional[dict] = None,
    boundaries: Optional[dict] = None,
    random: bool = False,
    mode: str = "space_modulation",
    verbose: bool = False,
    random_seed: Optional[int] = None,
) -> tuple[Setup, list]:
    """
    Define a quasi-universal interferometer (UIFO) as a grid of unit cells.

    The UIFO is constructed as:
    - An interior grid of unit cells (size x size).
    - Four boundaries around the grid, each boundary cell contains a source or readout.

    Parameters
    ----------
    size:
        Grid size. For example, size=3 creates a 3x3 interior grid.
    centers:
        Dictionary defining the centers of the unit cells.

        Keys are "xy" coordinates (as strings), values are tuples:
        (component_type, left_port_position)

        - component_type is one of:
          ["beamsplitter", "directional_beamsplitter"]
        - left_port_position is one of:
          ["left", "top", "right", "bottom"]

        Unspecified centers are filled with defaults or random choices (if random=True).
    boundaries:
        Dictionary defining boundary sources or readouts.

        Keys are "xy" coordinates on the boundary ring, values are strings:
        - "laser"
        - "squeezer"
        - "detector"
        - "balanced_homodyne"

        Unspecified boundaries are filled with defaults or random choices (if random=True).

        If random=True and neither "detector" nor "balanced_homodyne" is present, at least one
        is inserted at a random boundary position.
    random:
        If True, choose random center and boundary configurations (with a detector guaranteed).
    mode:
        Modulation mode:
        - "space_modulation"
        - "frequency_modulation"
        - "amplitude_modulation"
    verbose:
        If True, return the resolved centers and boundaries dictionaries as well.
    random_seed:
        Seed for reproducible random configuration. If None, a fresh seed is used.

    Returns
    -------
    setup:
        The setup object containing all components and their connections.
    component_property_pairs:
        The list of optimizable parameters of the setup.
    centers (only if verbose=True):
        The fully resolved centers mapping.
    boundaries (only if verbose=True):
        The fully resolved boundaries mapping.

    Raises
    ------
    ValueError:
        If `mode` is invalid.
    """
    rng = np.random.default_rng(random_seed)

    # Resolve center defaults
    if random:
        orientations = ["left", "top", "right", "bottom"]
        center_choices = ["beamsplitter", "directional_beamsplitter"]
        default_center_function: Callable[[], Tuple[str, str]] = lambda: (
            rng.choice(center_choices),
            rng.choice(orientations),
        )
    else:
        default_center_function = lambda: ("beamsplitter", "left")

    default_centers = defaultdict(default_center_function)
    default_centers.update(centers or {})
    centers = default_centers

    # Resolve boundary defaults
    if random:
        boundary_choices = ["laser", "squeezer"]
        default_boundary_function: Callable[[], str] = lambda: rng.choice(boundary_choices)
    else:
        default_boundary_function = lambda: "laser"

    default_boundaries = defaultdict(default_boundary_function)
    default_boundaries.update(boundaries or {})
    boundaries = default_boundaries

    # Ensure at least one detector or balanced homodyne readout if random=True
    if random:
        if "detector" not in default_boundaries.values() and "balanced_homodyne" not in default_boundaries.values():
            side = rng.choice([0, size + 1])
            position = rng.choice(range(1, size + 1))
            detector_type = rng.choice(["detector", "balanced_homodyne"])
            if rng.choice([True, False]):
                default_boundaries[f"{side}{position}"] = detector_type
            else:
                default_boundaries[f"{position}{side}"] = detector_type

    def unit_cell(
        S: Setup,
        x: int,
        y: int,
        center: str = "beamsplitter",
        left_port_position: str = "left",
    ) -> None:
        """
        Add a single interior unit cell at grid coordinate (x, y).

        The unit cell contains:
        - a central beamsplitter-like element (beamsplitter or directional_beamsplitter)
        - four surrounding mirrors (left, right, top, bottom)
        - free_mass suspension nodes for each mirror and for the central beamsplitter if present
        - spaces connecting the center to each mirror

        Signal nodes (only for space_modulation) are attached to each center-to-mirror space,
        with phases determined by physical direction:
        - horizontal spaces: phase 0
        - vertical spaces: phase 180
        """
        if center == "beamsplitter":
            S.add("beamsplitter", f"center{x}{y}")
            S.add("free_mass", f"center{x}{y}sus", target=f"center{x}{y}")
        elif center == "directional_beamsplitter":
            S.add("directional_beamsplitter", f"center{x}{y}")

        # Mirrors around the center: ml, mr, mt, mb
        S.add("mirror", f"ml{x}{y}")
        S.add("mirror", f"mr{x}{y}")
        S.add("mirror", f"mt{x}{y}")
        S.add("mirror", f"mb{x}{y}")

        # Free masses for each mirror
        S.add("free_mass", f"ml{x}{y}sus", target=f"ml{x}{y}")
        S.add("free_mass", f"mr{x}{y}sus", target=f"mr{x}{y}")
        S.add("free_mass", f"mt{x}{y}sus", target=f"mt{x}{y}")
        S.add("free_mass", f"mb{x}{y}sus", target=f"mb{x}{y}")

        # Orientation mapping: which beamsplitter ports correspond to which physical direction
        ports = {
            "left": ["left", "top", "right", "bottom"],
            "top": ["bottom", "left", "top", "right"],
            "right": ["right", "bottom", "left", "top"],
            "bottom": ["top", "right", "bottom", "left"],
        }

        mirrors = ["ml", "mt", "mr", "mb"]
        for i, port in enumerate(ports[left_port_position]):
            # Centers connect outward to mirrors, always using the mirror left port by default.
            S.space(f"center{x}{y}", f"{mirrors[i]}{x}{y}", length=1, source_port=port)

        if mode == "space_modulation":
            # Signals target edge names "centerxy_m?xy"
            S.add("signal", f"scenter{x}{y}ml{x}{y}", target=f"center{x}{y}_ml{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mr{x}{y}", target=f"center{x}{y}_mr{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mt{x}{y}", target=f"center{x}{y}_mt{x}{y}", phase=180)
            S.add("signal", f"scenter{x}{y}mb{x}{y}", target=f"center{x}{y}_mb{x}{y}", phase=180)

    def boundary_cell(
        S: Setup,
        x: int,
        y: int,
        boundary: str = "laser",
        mass: bool = True,
        position: str = "left",
    ) -> None:
        """
        Add a boundary cell at coordinate (x, y) outside the main interior grid.

        The boundary cell always includes:
        - a mirror node mxy
        - optionally a free_mass suspension for that mirror

        Depending on `boundary`, it also includes:
        - "detector": a detector and a qnoised readout at the mirror left port
        - "balanced_homodyne": a local oscillator laser, a beamsplitter, readout noise nodes, and detectors
        - "laser" or "squeezer": a source connected to the mirror

        Parameters
        ----------
        position:
            Indicates which side of the overall grid this boundary cell is on:
            "left", "right", "top", "bottom". Used to set signal phases in space_modulation mode.
        """
        S.add("mirror", f"m{x}{y}")

        if mass:
            S.add("free_mass", f"m{x}{y}sus", target=f"m{x}{y}")

        # Sources always connect to the mirror left port
        if boundary == "detector":
            S.add("detector", f"boundary{x}{y}detector", target=f"m{x}{y}", port="left", direction="out")
            S.add("qnoised", f"boundary{x}{y}noise", target=f"m{x}{y}", port="left", direction="out")

        elif boundary == "balanced_homodyne":
            # Local oscillator is not optimizable by default
            S.add("laser", f"boundary{x}{y}lo", power=0.01, phase=0, not_optimizable=True)
            S.add("beamsplitter", f"boundary{x}{y}bhbs")
            S.add("qnoised", f"boundary{x}{y}noise-top", target=f"boundary{x}{y}bhbs",
                  port="top", direction="out", auxiliary=True)
            S.add("qnoised", f"boundary{x}{y}noise-right", target=f"boundary{x}{y}bhbs",
                  port="right", direction="out", auxiliary=True)
            S.add("qhd", f"boundary{x}{y}noise", detector1=f"boundary{x}{y}noise-top",
                  detector2=f"boundary{x}{y}noise-right")
            S.add("detector", "detector-top", target=f"boundary{x}{y}bhbs", port="top", direction="out")
            S.add("detector", "detector-right", target=f"boundary{x}{y}bhbs", port="right", direction="out")

            # Connect local oscillator and mirror into the balanced homodyne beamsplitter
            S.space(f"boundary{x}{y}lo", f"boundary{x}{y}bhbs", target_port="bottom")
            S.space(f"m{x}{y}", f"boundary{x}{y}bhbs", source_port="left", target_port="left")

            if mode == "space_modulation":
                # Space signal phases depend on whether the space is vertical or horizontal
                S.add(
                    "signal",
                    f"sm{x}{y}boundary{x}{y}bhbs",
                    target=f"m{x}{y}_boundary{x}{y}bhbs",
                    phase=180 if position in ["top", "bottom"] else 0,
                )
                S.add(
                    "signal",
                    f"sboundary{x}{y}loboundary{x}{y}bhbs",
                    target=f"boundary{x}{y}lo_boundary{x}{y}bhbs",
                    phase=180 if position in ["left", "right"] else 0,
                )
            if mode == "amplitude_modulation":
                S.add(
                    "signal",
                    f"sboundary{x}{y}lo",
                    target=f"boundary{x}{y}lo_amplitude",
                    amplitude=(f"boundary{x}{y}lo_power", jnp.sqrt),
                )
            if mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}lo", target=f"boundary{x}{y}lo_frequency")

        elif boundary in ["laser", "squeezer"]:
            S.add(boundary, f"boundary{x}{y}")
            S.space(f"boundary{x}{y}", f"m{x}{y}")

            if mode == "space_modulation":
                # For vertical boundary connections (top and bottom), use phase 180
                S.add(
                    "signal",
                    f"sboundary{x}{y}m{x}{y}",
                    target=f"boundary{x}{y}_m{x}{y}",
                    phase=180 if position in ["top", "bottom"] else 0,
                )

            if boundary == "laser" and mode == "amplitude_modulation":
                S.add(
                    "signal",
                    f"sboundary{x}{y}",
                    target=f"boundary{x}{y}_amplitude",
                    amplitude=(f"boundary{x}{y}_power", jnp.sqrt),
                )
            if boundary == "laser" and mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}", target=f"boundary{x}{y}_frequency")

        else:
            raise ValueError(
                f"Invalid boundary component '{boundary}'. "
                "Choose from 'laser', 'squeezer', 'detector', 'balanced_homodyne'."
            )

    def cell_grid(S: Setup, n: int) -> None:
        """
        Build the interior grid of unit cells and their interconnections.

        Steps
        -----
        1) Create each unit cell and its internal connections.
        2) Create connections between neighboring cells:
           - Vertical: mt(x,y) connects to mb(x-1,y) for x > 1
           - Horizontal: mr(x,y-1) connects to ml(x,y) for y > 1
        """
        # Add unit cells
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                center, left_port_position = centers[f"{x}{y}"]
                unit_cell(S, x, y, center=center, left_port_position=left_port_position)

        # Connect unit cells to neighbors inside the grid
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                if x > 1:
                    # Use right ports because left ports are taken by center connections
                    S.space(f"mt{x}{y}", f"mb{x-1}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        S.add("signal", f"smt{x}{y}mb{x-1}{y}", target=f"mt{x}{y}_mb{x-1}{y}", phase=180)
                if y > 1:
                    S.space(f"mr{x}{y-1}", f"ml{x}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        S.add("signal", f"smr{x}{y-1}ml{x}{y}", target=f"mr{x}{y-1}_ml{x}{y}")

    S = Setup()
    S.add("frequency", "f")
    cell_grid(S, size)

    # Left and right boundaries
    for x in range(1, size + 1):
        boundary_cell(S, x, 0, boundary=boundaries[f"{x}0"], position="left")
        S.space(f"m{x}0", f"ml{x}1", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm{x}0ml{x}1", target=f"m{x}0_ml{x}1")

        boundary_cell(S, x, size + 1, boundary=boundaries[f"{x}{size+1}"], position="right")
        S.space(f"mr{x}{size}", f"m{x}{size+1}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smr{x}{size}m{x}{size+1}", target=f"mr{x}{size}_m{x}{size+1}")

    # Top and bottom boundaries
    for y in range(1, size + 1):
        boundary_cell(S, 0, y, boundary=boundaries[f"0{y}"], position="top")
        S.space(f"m0{y}", f"mt1{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm0{y}mt1{y}", target=f"m0{y}_mt1{y}", phase=180)

        boundary_cell(S, size + 1, y, boundary=boundaries[f"{size+1}{y}"], position="bottom")
        S.space(f"mb{size}{y}", f"m{size+1}{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smb{size}{y}m{size+1}{y}", target=f"mb{size}{y}_m{size+1}{y}", phase=180)

    if verbose:
        # Convert defaultdict values to plain serializable strings
        for key in default_centers:
            default_centers[key] = (str(default_centers[key][0]), str(default_centers[key][1]))
        for key in default_boundaries:
            default_boundaries[key] = str(default_boundaries[key])
        return S, S.parameters, default_centers, default_boundaries

    return S, S.parameters


# ----------------------------------------------------------------------
# Optimization constraint utilities
# ----------------------------------------------------------------------

def constrain_inter_grid_cell_spaces(
    component_property_pairs: List[ParameterPair],
    optimized_properties: List[str],
) -> list:
    """
    UIFO-specific parameter constraint helper.

    This function filters and groups parameters so that parallel spaces in different
    grid cells share the same optimization variable. This preserves the grid structure
    when optimizing lengths.

    The input `component_property_pairs` is typically `setup.parameters`.

    Behavior
    --------
    - First, parameters are filtered so only those whose property name is in `optimized_properties`
      remain.
    - Parameters with property "length" are treated specially:
      - Some interior connections are grouped:
        - Horizontal inter-cell connections (mr*_ml*) are grouped by a column index.
        - Vertical inter-cell connections (mt*_mb*) are grouped by a row index.
      - Spaces that involve "center" or "boundary" are excluded from length grouping.
    - Non-length parameters are left as individual entries.

    Parameters
    ----------
    component_property_pairs:
        List of (component_identifier, property_name) pairs.
    optimized_properties:
        List of property names that are actually optimized, for example ["length"].

    Returns
    -------
    constrained_pairs:
        A list containing a mix of:
        - single [component_identifier, property_name]
        - grouped lists of [component_identifier, property_name] entries

        If a group has only one element, it is flattened back to a single entry.

    Notes
    -----
    - This grouping logic assumes UIFO naming conventions:
      - mirrors named mlXY, mrXY, mtXY, mbXY
      - inter-cell edges named like "mr{x}{y-1}_ml{x}{y}" and "mt{x}{y}_mb{x-1}{y}"
    """
    # Keep only properties that are actually optimized
    component_property_pairs = [
        [component_name, property_name]
        for component_name, property_name in component_property_pairs
        if property_name in optimized_properties
    ]

    constrained_pairs: List[Any] = []

    # Two group dictionaries:
    # - index 0: horizontal spaces
    # - index 1: vertical spaces
    constrained_pair_dicts = [defaultdict(list), defaultdict(list)]

    for component_name, property_name in component_property_pairs:
        if property_name == "length":
            # Skip spaces that connect to centers or boundaries, keep them independent or ignore
            if "center" in component_name or "boundary" in component_name:
                continue

            # Horizontal inter-cell spaces: "...mr..._ml..."
            if "mr" in component_name and "_ml" in component_name:
                # Group key is derived from a coordinate character in the naming convention
                constrained_pair_dicts[0][component_name.split("_")[0][-1]].append([component_name, property_name])

            # Vertical inter-cell spaces: "...mt..._mb..."
            elif "mt" in component_name and "_mb" in component_name:
                constrained_pair_dicts[1][component_name.split("_")[0][-2]].append([component_name, property_name])

            else:
                constrained_pairs.append([component_name, property_name])

        else:
            constrained_pairs.append([component_name, property_name])

    # Append grouped constraints
    for constrained_pair_dict in constrained_pair_dicts:
        constrained_pairs.extend([constrained_pair_dict[key] for key in constrained_pair_dict])

    # Remove empty entries and flatten one-element groups
    constrained_pairs = [parameter for parameter in constrained_pairs if parameter]
    constrained_pairs = [parameter[0] if len(parameter) == 1 else parameter for parameter in constrained_pairs]
    return constrained_pairs
