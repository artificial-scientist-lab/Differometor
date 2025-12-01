import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from differometor.components import DEFAULT_PROPERTIES

class Nodes:
    def __init__(self, nodes):
        self._nodes = nodes

    def __iter__(self):
        return iter(self._nodes.items())
    
    def __call__(self, data=True):
        if data:
            return iter(self._nodes.items())
        else:
            return iter(self._nodes.keys())
    
    def __getitem__(self, node):
        if node in self._nodes:
            return self._nodes[node]
        else:
            raise KeyError(f"Node '{node}' not found in the setup.")


class Edges:
    def __init__(self, edges):
        self._edges = edges

    def __iter__(self):
        return iter((src, tgt, data) for (src, tgt), data in self._edges.items())
    
    def __getitem__(self, edge):
        source, target = edge.split('_')
        if (source, target) in self._edges:
            return self._edges[(source, target)]
        else:
            raise KeyError(f"Edge '{edge}' not found in the setup.")
        
    def __call__(self, data=True):
        if data:
            return iter((src, tgt, data) for (src, tgt), data in self._edges.items())
        else:
            return iter((src, tgt) for (src, tgt) in self._edges)


class Setup:
    def __init__(self):
        self.parameters = []
        self._nodes = {}
        self._edges = {}
        self.nodes = Nodes(self._nodes)
        self.edges = Edges(self._edges)        

        self.default_properties = DEFAULT_PROPERTIES

    def add(
            self, 
            component: str, 
            name: str, 
            not_optimizable=None, 
            target=None, 
            port=None, 
            direction=None, 
            auxiliary=None, 
            detector1=None,
            detector2=None,
            **properties):
        if '_' in name:
            raise ValueError(f"Node name '{name}' cannot contain underscores. Use '-' instead.")
        
        if not_optimizable is None:
            not_optimizable = []
        
        if component in ['mirror', 'beamsplitter']:
            if 'reflectivity' in properties and 'transmissivity' in properties:
                raise ValueError("Cannot specify both 'reflectivity' and 'transmissivity'. Use 'reflectivity' and 'loss' or 'transmissivity' and 'loss' instead.")
            if 'transmissivity' in properties:
                properties = {**self.default_properties[component], **properties}
                transmissivity = properties.pop('transmissivity')
                properties['reflectivity'] = (1 - transmissivity - properties['loss']) / (1 - properties['loss'])
            elif 'reflectivity' in properties:
                properties = {**self.default_properties[component], **properties}
                properties['reflectivity'] = properties['reflectivity'] / (1 - properties['loss'])

        try:
            properties = {**self.default_properties[component], **properties}
        except KeyError:
            raise ValueError(f"Component '{component}' is not recognized.")
        
        if len(properties) != len(self.default_properties[component]):
            raise ValueError(f"Component '{component}' has the properties {list(self.default_properties[component].keys())} but received {list(properties.keys())}.")
        if not component == 'signal' and not component == 'frequency':
            if not_optimizable is not True:
                self.parameters.extend([(name, property_name) for property_name in properties.keys() if (name, property_name) not in not_optimizable])

        self._nodes[name] = {
            'component': component,
            'properties': properties
        }

        if target is not None:
            if '_' in target:
                if target.split('_')[-1] not in ['amplitude', 'frequency']:
                    try:
                        self.edges[target]
                    except KeyError:
                        raise ValueError(f"Target '{target}' is not in the setup.")
                    self._nodes[name]['target'] = target
                else:
                    try:
                        self.nodes[target.split('_')[0]]
                    except KeyError:
                        raise ValueError(f"Target '{target.split('_')[0]}' is not in the setup.")
                    self._nodes[name]['target'] = target.split('_')[0]
                    self._nodes[name]['target_property'] = target.split('_')[-1]
            else:
                try:
                    self.nodes[target]
                except KeyError:
                    raise ValueError(f"Target '{target}' is not in the setup.")
                self._nodes[name]['target'] = target

        if port is not None:
            if port not in ['left', 'top', 'right', 'bottom']:
                raise ValueError(f"Port '{port}' is not recognized. Use 'left', 'top', 'right', or 'bottom'.")
            self._nodes[name]['port'] = port

        if direction is not None:
            if direction not in ['in', 'out']:
                raise ValueError(f"Direction '{direction}' is not recognized. Use 'in' or 'out'.")
            self._nodes[name]['direction'] = direction

        if auxiliary is not None:
            if auxiliary not in [True, False]:
                raise ValueError(f"Auxiliary '{auxiliary}' is not recognized. Use True or False.")
            self._nodes[name]['auxiliary'] = auxiliary

        if detector1 is not None:
            if detector1 not in self._nodes:
                raise ValueError(f"Detector1 '{detector1}' is not in the setup.")
            self._nodes[name]['detector1'] = detector1

        if detector2 is not None:
            if detector2 not in self._nodes:
                raise ValueError(f"Detector2 '{detector2}' is not in the setup.")
            self._nodes[name]['detector2'] = detector2

    def space(
            self, 
            source: str, 
            target: str, 
            not_optimizable=None, 
            source_port="right", 
            target_port="left", 
            **properties
        ):
        try:
            self._nodes[source]
        except KeyError:
            raise ValueError(f"Source node '{source}' is not in the setup.")
        try:
            self._nodes[target]
        except KeyError:
            raise ValueError(f"Target node '{target}' is not in the setup.")
        
        if not_optimizable is None:
            not_optimizable = []

        if properties is None:
            properties = {}
        properties = {**self.default_properties['space'], **properties}
        if len(properties) != len(self.default_properties['space']):
            raise ValueError(f"Space has the properties {list(self.default_properties['space'].keys())} but received {list(properties.keys())}.")
        if not_optimizable is not True:
            self.parameters.extend([(f"{source}_{target}", property_name) for property_name in properties.keys() if (f"{source}_{target}", property_name) not in not_optimizable])

        self._edges[(source, target)] = {
            'properties': properties,
            'source_port': source_port,
            'target_port': target_port
        }

    def to_data(
            self
        ) -> dict:
        """Return the setup as a plain Python dict with string keys for edges."""
        edges_serializable = {
            f"{src}_{tgt}": data for (src, tgt), data in self._edges.items()
        }
        return {
            "parameters": list(self.parameters),
            "nodes": dict(self._nodes),
            "edges": edges_serializable
        }

    @classmethod
    def from_data(
            cls,    
            data: dict
        ):
        """Rebuild a Setup instance from a dict with string edge keys."""
        setup = cls()
        setup.parameters = list(data["parameters"])
        setup._nodes = dict(data["nodes"])
        setup._edges = {
            tuple(edge_str.split("_", 1)): edge_data
            for edge_str, edge_data in data["edges"].items()
        }

        # Re-link wrappers
        setup.nodes = Nodes(setup._nodes)
        setup.edges = Edges(setup._edges)
        return setup


### Voyager setup


def voyager(
        mode = "space_modulation"
    ) -> tuple[Setup, list]:
    """
    Returns a predefined Voyager setup. 

    Parameters
    ----------
    mode: str 
        The modulation type. Available types: "space_modulation", "amplitude_modulation", "frequency_modulation"

    Returns
    -------
    setup: Setup
        The Voyager setup.
    parameters: list
        The parameters within the Voyager setup.
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

    S.add("free_mass", "prmsus", mass=29.243802983873618, target="prm")
    S.add("free_mass", "bssus", mass=48.634040943805395, target="bs")
    S.add("free_mass", "itmysus", mass=200, target="itmy")
    S.add("free_mass", "etmysus", mass=200, target="etmy")
    S.add("free_mass", "itmxsus", mass=200, target="itmx")
    S.add("free_mass", "etmxsus", mass=200, target="etmx")
    S.add("free_mass", "srmsus", mass=50, target="srm")

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

    S.add("frequency", "f", frequency=1)
    if mode == "space_modulation":
        S.add("signal", "fl0prm", target="l0_prm")
        S.add("signal", "fprmbs", target="prm_bs")
        S.add("signal", "fbsitmy", target="bs_itmy", phase=180)
        S.add("signal", "fitmyetmy", target="itmy_etmy", phase=180)
        S.add("signal", "fbsitmx", target="bs_itmx")
        S.add("signal", "fitmxetmx", target="itmx_etmx")
        S.add("signal", "bssrm", target="bs_srm", phase=180)
    elif mode == "amplitude_modulation":
        S.add("signal", "fl0", target="l0_amplitude", amplitude=("l0_power", jnp.sqrt))
        S.add("signal", "flo", target="lo_amplitude", amplitude=("lo_power", jnp.sqrt))
    elif mode == "frequency_modulation":
        S.add("signal", "fl0", target="l0_frequency")
        S.add("signal", "flo", target="lo_frequency")
    else:
        raise ValueError("Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'.")
    S.add("qnoised", "noise-top", target="bhbs", port="top", direction="out", auxiliary=True)
    S.add("qnoised", "noise-right", target="bhbs", port="right", direction="out", auxiliary=True)
    S.add("qhd", "noise", detector1="noise-top", detector2="noise-right", phase=180)
    S.add("detector", "detector-top", target="bhbs", port="top", direction="out")
    S.add("detector", "detector-right", target="bhbs", port="right", direction="out")

    return S, S.parameters


### Simplified aLIGO setup with optomechanics and squeezing


def aligo(
        mode="space_modulation"
    ) -> tuple[Setup, list]:
    """
    Returns a predefined aLIGO setup. 

    Parameters
    ----------
    mode: str 
        The modulation type. Available types: "space_modulation", "amplitude_modulation", "frequency_modulation"

    Returns
    -------
    setup: Setup
        The aLIGO setup.
    parameters: list
        The parameters within the aLIGO setup.
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

    S.add("free_mass", "itmxsus", mass=Mtm, target="itmx")
    S.add("free_mass", "etmxsus", mass=Mtm, target="etmx")
    S.add("free_mass", "itmysus", mass=Mtm, target="itmy")
    S.add("free_mass", "etmysus", mass=Mtm, target="etmy")

    S.space("L0", "prm")
    S.space("prm", "bs", length=53)
    S.space("bs", "itmx", length=4.5)
    S.space("itmx", "etmx", length=Larm)
    S.space("bs", "itmy", length=4.45, source_port="top")
    S.space("itmy", "etmy", length=Larm)
    S.space("bs", "srm", length=50.525, source_port="bottom")
    S.space("sq1", "srm", target_port="right")

    S.add("frequency", "f", frequency=5)
    
    if mode == "space_modulation":
        S.add("signal", "darmx", target="itmx_etmx")
        S.add("signal", "darmy", target="itmy_etmy", phase=180)
    elif mode == "frequency_modulation":
        S.add("signal", "fL0", target="L0_frequency")
    elif mode == "amplitude_modulation":
        S.add("signal", "fL0", target="L0_amplitude", amplitude=("L0_power", jnp.sqrt))
    else:
        raise ValueError("Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'.")

    S.add("qnoised", "noise", target="srm", port="right", direction="out")
    S.add("detector", "detector", target="srm", port="right", direction="out")

    return S, S.parameters


### UIFO


def uifo(
        size: int, 
        centers: dict = None, 
        boundaries: dict = None,
        random: bool = False,
        mode: str = 'space_modulation',
        verbose: bool = False,
        random_seed: int = None
    ) -> tuple[Setup, list]:
    """
    Defines a quasi-universal interferometer (UIFO).

    Parameters
    ----------
    size: int 
        The size of the grid. E.g. 3 results in a 3x3 grid.
    centers: dict
        A dictionary defining the centers of the unit cells. Keys are "xy" coordinates, values are 
        tuples of (component_type, left_port_position). By default, all centers are beamsplitters oriented
        towards the left. Component types are ["beamsplitter", "directional_beamsplitter"]. left_port_position
        should be one of ["left", "top", "right", "bottom"]. Unspecified centers will be filled with the 
        default or with a random component.
    boundaries: dict
        A dictionary defining the boundaries of the unit cells. Keys are "xy" coordinates, values are
        "component_type" strings. By default, all boundaries are lasers. Component types are ["laser", 
        "squeezer", "detector", "balanced_homodyne"]. Unspecified boundaries will be filled with the 
        default or with a random component.
    random: bool
        Whether to use random configurations for the centers and boundaries. The random configuration
        will have at least one detector.
    mode: str
        The modulation mode to use. Options are "space_modulation", "frequency_modulation", and
        "amplitude_modulation".
    verbose: bool
        Whether to return chosen centers and boundaries.
    random_seed: int | None
        The random seed for reproducibility. If None, a random random seed will be used.

    Returns
    -------
    setup: Setup
        The setup object containing all components and their connections.
    component_property_pairs: list
        A list of optimizable parameters of the setup. frequency, signal and the local oscillator in
        a balanced homodyne detection scheme are not optimizable by default.
    centers (if verbose): dict
        A dictionary defining the centers of the unit cells (see above).
    boundaries (if verbose): dict
        A dictionary defining the boundaries of the unit cells (see above).
    """
    rng = np.random.default_rng(random_seed)
    
    if random:
        orientations = ["left", "top", "right", "bottom"]
        center_choices = ["beamsplitter", "directional_beamsplitter"]
        default_center_function = lambda: (rng.choice(center_choices), rng.choice(orientations))
    else:
        default_center_function = lambda: ('beamsplitter', 'left')

    default_centers = defaultdict(default_center_function)
    default_centers.update(centers or {})
    centers = default_centers

    if random:
        boundary_choices = ["laser", "squeezer"]
        default_boundary_function = lambda: rng.choice(boundary_choices)
    else:
        default_boundary_function = lambda: 'laser'

    default_boundaries = defaultdict(default_boundary_function)
    default_boundaries.update(boundaries or {})
    boundaries = default_boundaries

    # Add "detector" to default_boundaries at a random position if not already present
    if random:
        if "detector" not in default_boundaries.values() and "balanced_homodyne" not in default_boundaries.values():
            side, position = rng.choice([0, size+1]), rng.choice(range(1, size+1))
            detector_type = rng.choice(["detector", "balanced_homodyne"])
            if rng.choice([True, False]):
                default_boundaries[f"{side}{position}"] = detector_type
            else:
                default_boundaries[f"{position}{side}"] = detector_type

    def unit_cell(S: Setup, x: int, y: int, center: str = "beamsplitter", left_port_position: str = "left"):
        if center == "beamsplitter":
            S.add("beamsplitter", f"center{x}{y}")
            S.add("free_mass", f"center{x}{y}sus", target=f"center{x}{y}")
        elif center == "directional_beamsplitter":
            S.add("directional_beamsplitter", f"center{x}{y}")

        S.add("mirror", f"ml{x}{y}")
        S.add("mirror", f"mr{x}{y}")
        S.add("mirror", f"mt{x}{y}")
        S.add("mirror", f"mb{x}{y}")
    
        S.add("free_mass", f"ml{x}{y}sus", target=f"ml{x}{y}")
        S.add("free_mass", f"mr{x}{y}sus", target=f"mr{x}{y}")
        S.add("free_mass", f"mt{x}{y}sus", target=f"mt{x}{y}")
        S.add("free_mass", f"mb{x}{y}sus", target=f"mb{x}{y}")

        ports = {
            "left":   ["left", "top", "right", "bottom"],
            "top":    ["bottom", "left", "top", "right"],
            "right":  ["right", "bottom", "left", "top"],
            "bottom": ["top", "right", "bottom", "left"]
        }

        mirrors = ["ml", "mt", "mr", "mb"]
        for i, port in enumerate(ports[left_port_position]):
            # centers always take the left side of unit cell mirrors
            S.space(f"center{x}{y}", f"{mirrors[i]}{x}{y}", length=1, source_port=port)    

        if mode == "space_modulation":
            # 0 phase for horizontal space, 180 for vertical spaces. 
            # Signal phase is based purely on geometry (physical direction of space):
            # This is independent of beamsplitter port orientation.
            S.add("signal", f"scenter{x}{y}ml{x}{y}", target=f"center{x}{y}_ml{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mr{x}{y}", target=f"center{x}{y}_mr{x}{y}", phase=0)
            S.add("signal", f"scenter{x}{y}mt{x}{y}", target=f"center{x}{y}_mt{x}{y}", phase=180)
            S.add("signal", f"scenter{x}{y}mb{x}{y}", target=f"center{x}{y}_mb{x}{y}", phase=180)

    def boundary_cell(S: Setup, x: int, y: int, boundary: str = "laser", mass: bool = True, position: str = "left"):
        S.add("mirror", f"m{x}{y}")
        
        if mass:
            S.add("free_mass", f"m{x}{y}sus", target=f"m{x}{y}")

        # sources always use left mirror port
        if boundary == "detector":
            S.add("detector", f"boundary{x}{y}detector", target=f"m{x}{y}", port="left", direction="out")
            S.add("qnoised", f"boundary{x}{y}noise", target=f"m{x}{y}", port="left", direction="out")
        elif boundary == "balanced_homodyne":
            S.add("laser", f"boundary{x}{y}lo", power=0.01, phase=0, not_optimizable=True)
            S.add("beamsplitter", f"boundary{x}{y}bhbs")
            S.add("qnoised", f"boundary{x}{y}noise-top", target=f"boundary{x}{y}bhbs", port="top", direction="out", auxiliary=True)
            S.add("qnoised", f"boundary{x}{y}noise-right", target=f"boundary{x}{y}bhbs", port="right", direction="out", auxiliary=True)
            S.add("qhd", f"boundary{x}{y}noise", detector1=f"boundary{x}{y}noise-top", detector2=f"boundary{x}{y}noise-right")
            S.add("detector", "detector-top", target=f"boundary{x}{y}bhbs", port="top", direction="out")
            S.add("detector", "detector-right", target=f"boundary{x}{y}bhbs", port="right", direction="out")

            S.space(f"boundary{x}{y}lo", f"boundary{x}{y}bhbs", target_port="bottom")
            S.space(f"m{x}{y}", f"boundary{x}{y}bhbs", source_port="left", target_port="left")
            if mode == "space_modulation":
                S.add("signal", f"sm{x}{y}boundary{x}{y}bhbs", target=f"m{x}{y}_boundary{x}{y}bhbs", phase=180 if position in ["top", "bottom"] else 0)
                S.add("signal", f"sboundary{x}{y}loboundary{x}{y}bhbs", target=f"boundary{x}{y}lo_boundary{x}{y}bhbs", phase=180 if position in ["left", "right"] else 0)
            if mode == "amplitude_modulation":
                S.add("signal", f"sboundary{x}{y}lo", target=f"boundary{x}{y}lo_amplitude", amplitude=(f"boundary{x}{y}lo_power", jnp.sqrt))
            if mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}lo", target=f"boundary{x}{y}lo_frequency")
        elif boundary in ["laser", "squeezer"]:
            S.add(boundary, f"boundary{x}{y}")
            S.space(f"boundary{x}{y}", f"m{x}{y}")
            if mode == "space_modulation":
                # phase 180 for signals on vertical spaces
                S.add("signal", f"sboundary{x}{y}m{x}{y}", target=f"boundary{x}{y}_m{x}{y}", phase=180 if position in ["top", "bottom"] else 0)
            if boundary == "laser" and mode == "amplitude_modulation":
                S.add("signal", f"sboundary{x}{y}", target=f"boundary{x}{y}_amplitude", amplitude=(f"boundary{x}{y}_power", jnp.sqrt))
            if boundary == "laser" and mode == "frequency_modulation":
                S.add("signal", f"sboundary{x}{y}", target=f"boundary{x}{y}_frequency")

    def cell_grid(S: Setup, n: int):
        for x in range(1, n+1):
            for y in range(1, n+1):
                center, left_port_position = centers[f"{x}{y}"]
                unit_cell(S, x, y, center=center, left_port_position=left_port_position)

        # connect individual unit cells inside the grid (not towards the boundaries)
        for x in range(1, n+1):
            for y in range(1, n+1):
                if x > 1:
                    # right ports because left ports are taken by center
                    S.space(f"mt{x}{y}", f"mb{x-1}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        # signals on vertical spaces
                        S.add("signal", f"smt{x}{y}mb{x-1}{y}", target=f"mt{x}{y}_mb{x-1}{y}", phase=180)
                if y > 1:
                    # right ports because left ports are taken by center
                    S.space(f"mr{x}{y-1}", f"ml{x}{y}", source_port="right", target_port="right")
                    if mode == "space_modulation":
                        # signals on horizontal spaces
                        S.add("signal", f"smr{x}{y-1}ml{x}{y}", target=f"mr{x}{y-1}_ml{x}{y}")

    S = Setup()
    S.add("frequency", "f")
    cell_grid(S, size)

    for x in range(1, size+1):
        # left boundary
        boundary_cell(S, x, 0, boundary=boundaries[f"{x}0"], position="left")
        # target port right because left is taken by center
        S.space(f"m{x}0", f"ml{x}1", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm{x}0ml{x}1", target=f"m{x}0_ml{x}1")
        # right boundary
        boundary_cell(S, x, size+1, boundary=boundaries[f"{x}{size+1}"], position="right")
        # mirrors on the right side of the grid have their right ports still open, 
        # boundary mirrors also only have their right ports open as sources always use left port
        S.space(f"mr{x}{size}", f"m{x}{size+1}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smr{x}{size}m{x}{size+1}", target=f"mr{x}{size}_m{x}{size+1}")
    for y in range(1, size+1):
        # top boundary
        boundary_cell(S, 0, y, boundary=boundaries[f"0{y}"], position="top")
        # mirrors along the top of the grid have their left ports towards the center, so only the right ports are open
        S.space(f"m0{y}", f"mt1{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"sm0{y}mt1{y}", target=f"m0{y}_mt1{y}", phase=180)
        # bottom boundary
        boundary_cell(S, size+1, y, boundary=boundaries[f"{size+1}{y}"], position="bottom")
        # mirrors along the bottom of the grid have their left ports towards the center, so only the right ports are open
        # boundary mirrors also only have their right ports open as sources always use left port
        S.space(f"mb{size}{y}", f"m{size+1}{y}", target_port="right")
        if mode == "space_modulation":
            S.add("signal", f"smb{size}{y}m{size+1}{y}", target=f"mb{size}{y}_m{size+1}{y}", phase=180)

    if verbose:
        for key in default_centers:
            default_centers[key] = (str(default_centers[key][0]), str(default_centers[key][1]))
        for key in default_boundaries:
            default_boundaries[key] = str(default_boundaries[key])
        return S, S.parameters, default_centers, default_boundaries
    else:
        return S, S.parameters


def constrain_inter_grid_cell_spaces(
        component_property_pairs, 
        optimized_properties
    ):
    """
    Used for UIFOs. Filters parameters that are actually getting optimized. Correlates parallel 
    spaces in different grid cells of an UIFO, so that grid structure is preserved. 

    Parameters
    ----------
    component_property_pairs : list
        A list of component-property pairs to constrain. E.g. [("laser1", "power"), 
        ("laser1", "phase"), ("mirror1", "reflectivity"), ...]
    optimized_properties : list
        A list of properties to optimize. E.g. ["power"]

    Returns
    -------
    constrained_pairs : list
        A list of constrained component-property pairs. E.g. [("laser1", "power"), 
        [("mirror1_mirror2", "length"), ("mirror3_mirror4", "length")]]
    """
    component_property_pairs = [[component_name, property_name] for component_name, property_name in component_property_pairs if property_name in optimized_properties]

    constrained_pairs = []
    # horizontal spaces and vertical spaces
    constrained_pair_dicts = [defaultdict(list), defaultdict(list)]
    for component_name, property_name in component_property_pairs:
        if property_name == "length":
            if "center" in component_name or "boundary" in component_name:
                continue
            elif "mr" in component_name and "_ml" in component_name:
                constrained_pair_dicts[0][component_name.split('_')[0][-1]].append([component_name, property_name])
            elif "mt" in component_name and "_mb" in component_name:
                constrained_pair_dicts[1][component_name.split('_')[0][-2]].append([component_name, property_name])
            else:
                constrained_pairs.append([component_name, property_name])
        else:
            constrained_pairs.append([component_name, property_name])
    for constrained_pair_dict in constrained_pair_dicts:
        constrained_pairs.extend([constrained_pair_dict[key] for key in constrained_pair_dict])

    constrained_pairs = [parameter for parameter in constrained_pairs if parameter]
    constrained_pairs = [parameter[0] if len(parameter) == 1 else parameter for parameter in constrained_pairs]
    return constrained_pairs


### Finesse Conversions


def differometor_to_finesse(
        setup: Setup
    ) -> str:
    finesse = ""

    port_translation = {
        "mirror": {
            "left": "p1",
            "right": "p2"
        },
        "laser": {
            "left": "p1",
            "right": "p1"
        },
        "squeezer": {
            "left": "p1",
            "right": "p1"
        },
        "beamsplitter": {
            "left": "p1",
            "top": "p2",
            "right": "p3",
            "bottom": "p4"
        },
        "directional_beamsplitter": {
            "left": "p1",
            "top": "p2",
            "right": "p3",
            "bottom": "p4"
        },
        "nothing": {
            "left": "p1",
            "right": "p2"
        }
    }

    direction_translation = {
        "in": "i",
        "out": "o"
    }

    for node, data in setup.nodes(data=True):
        try:
            property_dict = DEFAULT_PROPERTIES[data["component"]].copy()
        except KeyError:
            print("KeyError for :", node, data)
        property_dict.update(data.get("properties", {}))
        
        if data["component"] == "frequency":
            finesse += f"fsig({property_dict['frequency']})\n"
        elif data["component"] == "laser":
            finesse += f"l {node} P={property_dict['power']} phase={property_dict['phase']}\n"
        elif data["component"] == "squeezer":
            finesse += f"sq {node} db={property_dict['db']} angle={property_dict['angle']}\n"
        elif data["component"] == "mirror":
            l = property_dict['loss']
            r = property_dict['reflectivity'] * (1 - l)
            t = 1 - r - l
            finesse += f"m {node} R={r} T={t} L={l} phi={property_dict['tuning']}\n"
        elif data["component"] == "beamsplitter":
            l = property_dict['loss']
            r = property_dict['reflectivity'] * (1 - l)
            t = 1 - r - l
            finesse += f"bs {node} R={r} T={t} L={l} phi={property_dict['tuning']} alpha={property_dict['alpha']}\n"
        elif data["component"] == "free_mass":
            finesse += f"free_mass {node} {data['target']}.mech mass={property_dict['mass']}\n"
        elif data["component"] == "signal":
            if "target_property" in data and data["target_property"] == "frequency":
                finesse += f"sgen {node} {data['target']}.frq.i amplitude={property_dict['amplitude']} phase={property_dict['phase']}\n"
            elif "target_property" in data and data["target_property"] == "amplitude":
                finesse += f"sgen {node} {data['target']}.amp.i amplitude=sqrt({data['target']}.P) phase={property_dict['phase']}\n"
            else:
                finesse += f"sgen {node} {data['target']}.h amplitude={property_dict['amplitude']} phase={property_dict['phase']}\n"
        elif data["component"] == "qnoised":
            port = port_translation[setup.nodes[data["target"]]["component"]][data["port"]]
            direction = direction_translation[data["direction"]]
            finesse += f"qnoised {node} {data['target']}.{port}.{direction}\n"
        elif data["component"] == "directional_beamsplitter":
            finesse += f"dbs {node}\n"
        elif data["component"] == "nothing":
            finesse += f"nothing {node}\n"
        
    finesse += "\n"
    for source, target, data in setup.edges(data=True):
        property_dict = DEFAULT_PROPERTIES["space"].copy()
        property_dict.update(data.get("properties", {}))
        source_port = port_translation[setup.nodes[source]["component"]][data.get("source_port", "right")]
        target_port = port_translation[setup.nodes[target]["component"]][data.get("target_port", "left")]
        direction = direction_translation[data.get("direction", "out")]
        finesse += f"s {source}_{target} {source}.{source_port} {target}.{target_port} L={property_dict['length']} nr={property_dict['refractive_index']}\n"
    return finesse
