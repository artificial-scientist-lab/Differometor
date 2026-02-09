"""
`visualize_setup` (Plotly HTML rendering) is intentionally split into stages.
If you need to extend behavior, follow this order:

1. Graph + placement:
   `_collect_nodes_to_skip` -> `_build_adjacency_and_attachments` ->
   `_place_connected_nodes` -> `_place_edge_attached_nodes` -> `_reposition_detectors`.
2. Beam metrics + scaling:
   `_prepare_beam_source_config` -> `_build_beam_records` ->
   `_global_max_series_by_source` -> `_add_beam_traces`.
3. Component geometry + rendering:
   `_collect_component_trace_data` -> `_add_component_traces`.
4. Interactive controls:
   `_build_controls` and final `fig.update_layout(...)`.

Important invariants for contributors:
- Keep `legendgroup` consistent across visible and invisible helper traces so legend toggling
  behaves correctly for all component types.
- Beam hover traces are separate from beam line traces; both must be updated together in
  `_beam_trace_updates`.
- Component hover priority is provided by the final invisible component-hover trace in
  `_add_component_traces`; avoid removing it unless replacing with an equivalent mechanism.
- `port_to_index` keys are assumed to match `"component.port.direction"` format.
"""

import os
from dataclasses import dataclass
import plotly.graph_objects as go
from collections import defaultdict, deque
import numpy as np


_CARDINAL_DIRECTIONS = ("left", "top", "right", "bottom")
_DIRECTION_VECTORS = {
    "left": np.array([-1.0, 0.0]),
    "top": np.array([0.0, 1.0]),
    "right": np.array([1.0, 0.0]),
    "bottom": np.array([0.0, -1.0]),
}
_PORT_STEPS = {"left": 0, "top": 1, "right": 2, "bottom": 3}
_HIDDEN_COMPONENTS = {"free_mass", "frequency", "qhd", "signal"}
_DETECTOR_LIKE_COMPONENTS = {"detector", "qnoised"}
_POWER_CARRIER_COMPONENTS = {"mirror", "beamsplitter", "directional_beamsplitter"}
_START_NODE_TYPES = {"laser", "squeezer", "detector", "qnoised"}


@dataclass
class _BeamSourceConfig:
    powers_array: np.ndarray | None
    signals_array: np.ndarray | None
    freq_values: np.ndarray | None
    source_order: list[str]
    frame_count_by_source: dict[str, int]
    initial_source: str
    initial_frame_by_source: dict[str, int]


@dataclass
class _BeamRecord:
    source: str
    target: str
    source_component: str
    target_component: str
    x: list[float]
    y: list[float]
    length: float
    metric_map: dict[str, dict[str, np.ndarray]]
    max_series: dict[str, np.ndarray]
    hover_point_count: int = 0


def _rotate_direction(direction, steps=1):
    idx = _CARDINAL_DIRECTIONS.index(direction)
    return _CARDINAL_DIRECTIONS[(idx + steps) % 4]


def _opposite_direction(direction):
    return _rotate_direction(direction, 2)


def _port_direction(left_dir, port):
    step = _PORT_STEPS.get(port, 0)
    return _rotate_direction(left_dir, step)


def _infer_left_direction(port, port_dir):
    step = _PORT_STEPS.get(port, 0)
    return _rotate_direction(port_dir, -step)


def _distance_for_space(length_value, min_space, length_scale):
    return min_space + length_scale * float(length_value)


def _rotate_vec(vec, angle_deg):
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _add_connection(adjacency, node_a, node_b, port_a, port_b, distance, kind):
    adjacency[node_a].append(
        {
            "neighbor": node_b,
            "port": port_a,
            "neighbor_port": port_b,
            "distance": distance,
            "kind": kind,
        }
    )
    adjacency[node_b].append(
        {
            "neighbor": node_a,
            "port": port_b,
            "neighbor_port": port_a,
            "distance": distance,
            "kind": kind,
        }
    )


def _node_hover_text(node_name, data, component_name):
    props = data.get("properties", {}) or {}
    lines = [f"<b>{node_name}</b>", f"component: {component_name}"]
    for key in sorted(props.keys()):
        lines.append(f"{key}: {props[key]}")
    return "<br>".join(lines)


def _beam_direction_for(node, adjacency, left_dirs):
    connections = adjacency.get(node, [])
    if not connections:
        return "right"
    port = connections[0]["port"]
    left_dir = left_dirs.get(node, "left")
    return _port_direction(left_dir, port)


def _component_hover_size(component_name):
    if component_name in {"beamsplitter", "directional_beamsplitter"}:
        return 28.0
    if component_name in {"laser", "squeezer"}:
        return 32.0
    if component_name == "detector":
        return 26.0
    if component_name == "mirror":
        return 24.0
    return 20.0


def _beam_width_from_scale(normalized_power, min_beam_width, width_scale, max_beam_width):
    width = min_beam_width + float(width_scale) * float(normalized_power)
    if max_beam_width is not None:
        width = min(width, float(max_beam_width))
    if not np.isfinite(width):
        width = min_beam_width
    return float(width)


def _beam_hover_size_from_width(width):
    # Keep hover targets large enough for usability but bounded to avoid
    # stealing hover from nearby component traces.
    return float(max(10.0, min(16.0, width + 5.0)))


def _collect_nodes_to_skip(setup, node_components):
    nodes_to_skip = {name for name, comp in node_components.items() if comp in _HIDDEN_COMPONENTS}
    detector_groups = defaultdict(list)
    for node, data in setup.nodes(data=True):
        if node_components.get(node) not in _DETECTOR_LIKE_COMPONENTS:
            continue
        target = data.get("target")
        if target is None:
            continue
        port = data.get("port", "left")
        direction = data.get("direction", "in")
        detector_groups[(target, port, direction)].append(node)

    for group_nodes in detector_groups.values():
        rep = next((n for n in group_nodes if node_components.get(n) == "detector"), group_nodes[0])
        for node in group_nodes:
            if node != rep:
                nodes_to_skip.add(node)
    return nodes_to_skip


def _build_adjacency_and_attachments(
    setup,
    node_components,
    nodes_to_skip,
    min_space,
    length_scale,
    attachment_length,
):
    adjacency = defaultdict(list)
    edge_attached_nodes = {}

    for source, target, data in setup.edges(data=True):
        if source in nodes_to_skip or target in nodes_to_skip:
            continue
        length_value = data.get("properties", {}).get("length", 0.0)
        _add_connection(
            adjacency,
            source,
            target,
            data.get("source_port", "right"),
            data.get("target_port", "left"),
            _distance_for_space(length_value, min_space, length_scale),
            "space",
        )

    for node, data in setup.nodes(data=True):
        if node in nodes_to_skip:
            continue
        target = data.get("target")
        if target is None:
            continue
        if target in node_components and target not in nodes_to_skip:
            _add_connection(
                adjacency,
                node,
                target,
                "left",
                data.get("port", "left"),
                attachment_length,
                "target",
            )
            continue
        if "_" in str(target):
            try:
                setup.edges[target]
                edge_attached_nodes[node] = target
            except KeyError:
                continue

    return adjacency, edge_attached_nodes


def _place_connected_nodes(setup, adjacency, node_components, nodes_to_skip, edge_attached_nodes, rng, min_space):
    nodes_to_place = {
        name
        for name, _ in setup.nodes(data=True)
        if name not in nodes_to_skip and name not in edge_attached_nodes
    }
    positions = {}
    left_dirs = {}
    component_spacing = 4.0 * min_space
    offset_x = 0.0

    while nodes_to_place:
        candidates = [name for name in nodes_to_place if node_components.get(name) in _START_NODE_TYPES]
        start_node = rng.choice(candidates or list(nodes_to_place))

        positions[start_node] = np.array([offset_x, 0.0])
        start_connections = adjacency.get(start_node, [])
        if start_connections:
            left_dirs[start_node] = _infer_left_direction(start_connections[0]["port"], "right")
        else:
            left_dirs[start_node] = "left"

        queue = deque([start_node])
        nodes_to_place.remove(start_node)
        current_component = {start_node}

        while queue:
            node = queue.popleft()
            node_pos = positions[node]
            node_left_dir = left_dirs.get(node, "left")

            for connection in adjacency.get(node, []):
                neighbor = connection["neighbor"]
                port = connection["port"]
                neighbor_port = connection["neighbor_port"]
                distance = connection["distance"]

                port_dir = _port_direction(node_left_dir, port)
                delta = _DIRECTION_VECTORS[port_dir] * distance
                neighbor_pos = node_pos + delta

                if neighbor not in positions:
                    positions[neighbor] = neighbor_pos
                    desired_dir = _opposite_direction(port_dir)
                    left_dirs[neighbor] = _infer_left_direction(neighbor_port, desired_dir)
                    if neighbor in nodes_to_place:
                        nodes_to_place.remove(neighbor)
                    queue.append(neighbor)
                    current_component.add(neighbor)
                elif neighbor not in left_dirs:
                    desired_dir = _opposite_direction(port_dir)
                    left_dirs[neighbor] = _infer_left_direction(neighbor_port, desired_dir)

        xs = [positions[node][0] for node in current_component]
        if xs:
            min_x = min(xs)
            max_x = max(xs)
            shift = offset_x - min_x
            if abs(shift) > 1e-9:
                for node in current_component:
                    positions[node][0] += shift
                max_x += shift
            offset_x = max_x + component_spacing

    return positions, left_dirs, offset_x, component_spacing


def _place_edge_attached_nodes(edge_attached_nodes, positions, left_dirs, min_space, component_spacing, offset_x):
    for node, edge_name in edge_attached_nodes.items():
        try:
            source, target = edge_name.split("_", 1)
        except ValueError:
            continue
        if source in positions and target in positions:
            p0 = positions[source]
            p1 = positions[target]
            midpoint = 0.5 * (p0 + p1)
            delta = p1 - p0
            if np.linalg.norm(delta) > 1e-6:
                normal = np.array([-delta[1], delta[0]])
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0.0, 1.0])
            positions[node] = midpoint + normal * (0.5 * min_space)
            left_dirs[node] = "left"
        else:
            positions[node] = np.array([offset_x, 0.0])
            left_dirs[node] = "left"
            offset_x += component_spacing
    return offset_x


def _space_ports(setup):
    ports = set()
    for source, target, data in setup.edges(data=True):
        ports.add((source, data.get("source_port", "right")))
        ports.add((target, data.get("target_port", "left")))
    return ports


def _reposition_detectors(setup, positions, left_dirs, node_components, nodes_to_skip, attachment_length):
    space_port_map = _space_ports(setup)
    for node, data in setup.nodes(data=True):
        if node in nodes_to_skip or node not in positions:
            continue
        if node_components.get(node) not in _DETECTOR_LIKE_COMPONENTS:
            continue

        target = data.get("target")
        if target not in positions or target not in node_components:
            continue
        if node_components.get(target) not in _POWER_CARRIER_COMPONENTS:
            continue

        port = data.get("port", "left")
        if (target, port) not in space_port_map:
            continue

        target_left_dir = left_dirs.get(target, "left")
        port_dir = _port_direction(target_left_dir, port)
        base_vec = _DIRECTION_VECTORS[port_dir]
        target_pos = positions[target]

        candidates = [
            target_pos + _rotate_vec(base_vec, 45.0) * attachment_length,
            target_pos + _rotate_vec(base_vec, -45.0) * attachment_length,
        ]

        other_positions = [pos for name, pos in positions.items() if name not in {node, target}]
        if other_positions:
            scores = [
                min(float(np.linalg.norm(candidate - other)) for other in other_positions)
                for candidate in candidates
            ]
            best_idx = int(np.argmax(scores))
        else:
            best_idx = 0
        positions[node] = candidates[best_idx]


def _prepare_beam_source_config(powers, signals, frequencies, signal_index):
    powers_array = None if powers is None else np.asarray(powers)
    signals_array = None if signals is None else np.asarray(signals)
    has_power_mode = powers_array is not None
    has_signal_mode = signals_array is not None

    signal_frame_count = 1
    freq_values = None
    if has_signal_mode:
        if signals_array.ndim < 2:
            raise ValueError("signals must have shape (port_number, n).")
        signal_frame_count = int(signals_array.shape[1])
        if signal_frame_count < 1:
            raise ValueError("signals must contain at least one frame along axis 1.")
        if frequencies is not None:
            freq_values = np.asarray(frequencies).reshape(-1)
            if freq_values.shape[0] != signal_frame_count:
                raise ValueError(
                    f"frequencies length ({freq_values.shape[0]}) must match signals.shape[1] ({signal_frame_count})."
                )

    source_order = []
    if has_power_mode:
        source_order.append("powers")
    if has_signal_mode:
        source_order.append("signals")
    if not source_order:
        source_order = ["powers"]

    frame_count_by_source = {"powers": 1, "signals": signal_frame_count}
    initial_source = "powers" if "powers" in source_order else "signals"
    initial_signal_index = int(np.clip(int(signal_index), 0, signal_frame_count - 1))
    initial_frame_by_source = {"powers": 0, "signals": initial_signal_index}
    return _BeamSourceConfig(
        powers_array=powers_array,
        signals_array=signals_array,
        freq_values=freq_values,
        source_order=source_order,
        frame_count_by_source=frame_count_by_source,
        initial_source=initial_source,
        initial_frame_by_source=initial_frame_by_source,
    )


def _port_metric_series(port_to_index, component_name, port_name, source_name, source_config):
    if not isinstance(port_to_index, dict):
        return {}

    values = {}
    for direction in ("in", "out"):
        key = f"{component_name}.{port_name}.{direction}"
        idx = port_to_index.get(key)
        if idx is None:
            continue
        try:
            if source_name == "signals":
                if source_config.signals_array is None:
                    continue
                selected = np.asarray(np.abs(source_config.signals_array[int(idx)]), dtype=float)
                if selected.ndim == 0:
                    series = np.array([float(selected)], dtype=float)
                elif selected.ndim == 1:
                    series = selected.astype(float, copy=False)
                else:
                    axes = tuple(range(1, selected.ndim))
                    series = np.nanmax(selected, axis=axes).astype(float, copy=False)
                if series.shape[0] != source_config.frame_count_by_source["signals"]:
                    continue
            else:
                if source_config.powers_array is None:
                    continue
                selected = np.asarray(source_config.powers_array[int(idx)], dtype=float)
                if selected.size == 0:
                    continue
                scalar = float(np.nanmax(np.abs(selected)))
                series = np.array([scalar], dtype=float)
        except (ValueError, TypeError, IndexError):
            continue
        values[key] = series
    return values


def _build_beam_records(setup, positions, node_components, port_to_index, source_config):
    """
    Build per-space beam records used for both rendering and hover text.

    Each record stores:
    - geometry (`x`, `y`, `length`)
    - endpoint component types (used for hover clearances)
    - per-source metric maps (`powers` / `signals`) and max-series vectors.

    Extension point:
    - Add new per-space metadata here if you want it available in both trace
      styling and hover text generation.
    """
    beam_records = []
    for source, target, data in setup.edges(data=True):
        if source not in positions or target not in positions:
            continue

        source_port = data.get("source_port", "right")
        target_port = data.get("target_port", "left")
        source_component = node_components.get(source, "")
        target_component = node_components.get(target, "")

        metric_map = {}
        max_series = {}
        for source_name in source_config.source_order:
            connected_metric_map = {}
            if source_component in _POWER_CARRIER_COMPONENTS:
                connected_metric_map.update(
                    _port_metric_series(port_to_index, source, source_port, source_name, source_config)
                )
            if target_component in _POWER_CARRIER_COMPONENTS:
                connected_metric_map.update(
                    _port_metric_series(port_to_index, target, target_port, source_name, source_config)
                )

            frame_count = source_config.frame_count_by_source[source_name]
            if connected_metric_map:
                stacked = np.vstack([v for v in connected_metric_map.values()])
                source_max = np.nanmax(stacked, axis=0)
            else:
                source_max = np.full((frame_count,), np.nan, dtype=float)
            metric_map[source_name] = connected_metric_map
            max_series[source_name] = source_max

        p0 = positions[source]
        p1 = positions[target]
        beam_records.append(
            _BeamRecord(
                source=source,
                target=target,
                source_component=source_component,
                target_component=target_component,
                x=[float(p0[0]), float(p1[0])],
                y=[float(p0[1]), float(p1[1])],
                length=float(data.get("properties", {}).get("length", 0.0)),
                metric_map=metric_map,
                max_series=max_series,
            )
        )
    return beam_records


def _global_max_series_by_source(beam_records, source_config):
    global_max = {}
    for source_name in source_config.source_order:
        frame_count = source_config.frame_count_by_source[source_name]
        if not beam_records:
            global_max[source_name] = np.zeros((frame_count,), dtype=float)
            continue

        stacked = np.vstack([record.max_series[source_name] for record in beam_records])
        finite = stacked[np.isfinite(stacked)]
        scalar_max = float(np.max(finite)) if finite.size else 0.0
        # Keep one normalization scale across all frames so signal slider
        # reflects absolute magnitude changes with frequency.
        values = np.full((frame_count,), scalar_max, dtype=float)
        global_max[source_name] = values
    return global_max


def _beam_hover_text(record, source_name, frame_idx, source_config):
    frame_count = source_config.frame_count_by_source[source_name]
    frame_idx = int(np.clip(frame_idx, 0, frame_count - 1))
    lines = [
        f"<b>{record.source} -> {record.target}</b>",
        "component: space",
        f"length: {record.length}",
    ]
    if source_name == "signals":
        if source_config.freq_values is not None:
            lines.append(f"frequency: {source_config.freq_values[frame_idx]}")
        else:
            lines.append(f"frequency index: {frame_idx}")
        lines.append("port signals:")
    else:
        lines.append("port powers:")

    metric_map = record.metric_map.get(source_name, {})
    max_series = record.max_series.get(source_name, np.array([np.nan]))
    if metric_map:
        for key in sorted(metric_map.keys()):
            value = float(metric_map[key][frame_idx])
            lines.append(f"{key}: {value:.6g}")
        lines.append(f"max used for width: {float(max_series[frame_idx]):.6g}")
    else:
        lines.append("unavailable")
    return "<br>".join(lines)


def _beam_width_for_record(record, source_name, frame_idx, global_max_series_by_source, min_beam_width, width_scale, max_beam_width):
    frame_global_max = float(global_max_series_by_source[source_name][frame_idx])
    record_max = float(record.max_series[source_name][frame_idx])
    if np.isfinite(record_max) and frame_global_max > 0:
        normalized = record_max / frame_global_max
    else:
        normalized = 0.0
    return _beam_width_from_scale(normalized, min_beam_width, width_scale, max_beam_width)


def _component_hover_clearance(component_name, min_space, source_length):
    if component_name in {"laser", "squeezer"}:
        return 0.5 * source_length + 0.05 * min_space
    if component_name == "detector":
        return 0.20 * min_space
    if component_name in {"mirror", "beamsplitter", "directional_beamsplitter"}:
        return 0.16 * min_space
    return 0.10 * min_space


def _mirror_half_extents(beam_dir, mirror_length, mirror_thickness):
    if beam_dir in {"left", "right"}:
        return mirror_thickness / 2, mirror_length / 2
    return mirror_length / 2, mirror_thickness / 2


def _add_beam_traces(
    fig,
    beam_records,
    source_config,
    global_max_by_source,
    min_space,
    source_length,
    min_beam_width,
    width_scale,
    max_beam_width,
):
    beam_legend_added = False
    beam_line_indices = []
    beam_hover_indices = []
    source_name = source_config.initial_source
    frame_idx = source_config.initial_frame_by_source[source_name]

    for record in beam_records:
        beam_width = _beam_width_for_record(
            record,
            source_name,
            frame_idx,
            global_max_by_source,
            min_beam_width,
            width_scale,
            max_beam_width,
        )
        hover_marker_size = _beam_hover_size_from_width(beam_width)
        hover_text = _beam_hover_text(record, source_name, frame_idx, source_config)

        fig.add_trace(
            go.Scatter(
                x=record.x,
                y=record.y,
                mode="lines",
                line=dict(color="red", width=beam_width),
                name="beam",
                legendgroup="beam",
                showlegend=not beam_legend_added,
                hoverinfo="skip",
            )
        )
        beam_line_indices.append(len(fig.data) - 1)
        beam_legend_added = True

        # Use invisible, dense center-segment markers to keep space hover
        # available along the beam while preserving endpoint component hover.
        p0 = np.array([record.x[0], record.y[0]], dtype=float)
        p1 = np.array([record.x[1], record.y[1]], dtype=float)
        distance = float(np.linalg.norm(p1 - p0))
        if distance <= 1e-12:
            continue
        start_clearance = _component_hover_clearance(record.source_component, min_space, source_length)
        end_clearance = _component_hover_clearance(record.target_component, min_space, source_length)
        t0 = max(0.125, start_clearance / distance)
        t1 = min(0.875, 1.0 - end_clearance / distance)
        if t1 <= t0:
            center = 0.5 * (t0 + t1)
            t0 = max(0.45, center - 0.02)
            t1 = min(0.55, center + 0.02)
            if t1 <= t0:
                t0, t1 = 0.48, 0.52
        n_hover_points = max(12, int(np.ceil(distance / max(min_space, 1e-9) * 24)))
        t = np.linspace(t0, t1, n_hover_points)
        hover_points = p0[None, :] + (p1 - p0)[None, :] * t[:, None]
        fig.add_trace(
            go.Scatter(
                x=hover_points[:, 0].tolist(),
                y=hover_points[:, 1].tolist(),
                mode="markers",
                marker=dict(size=hover_marker_size, color="rgba(0,0,0,0)", line=dict(width=0)),
                legendgroup="beam",
                showlegend=False,
                text=[hover_text] * n_hover_points,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        beam_hover_indices.append(len(fig.data) - 1)
        record.hover_point_count = n_hover_points

    return beam_line_indices, beam_hover_indices


def _add_attachment_trace(fig, setup, positions, nodes_to_skip):
    attach_x, attach_y = [], []
    for node, data in setup.nodes(data=True):
        if node in nodes_to_skip:
            continue
        target = data.get("target")
        if target is None:
            continue
        if target in positions and node in positions:
            p0 = positions[node]
            p1 = positions[target]
            attach_x.extend([p0[0], p1[0], None])
            attach_y.extend([p0[1], p1[1], None])
    if attach_x:
        fig.add_trace(
            go.Scatter(
                x=attach_x,
                y=attach_y,
                mode="lines",
                line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dash"),
                name="attachment",
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_segment(trace, p0, p1, hover):
    trace["x"].extend([p0[0], p1[0], None])
    trace["y"].extend([p0[1], p1[1], None])
    trace["text"].extend([hover, hover, None])


def _collect_component_trace_data(
    setup,
    positions,
    left_dirs,
    adjacency,
    nodes_to_skip,
    min_space,
    mirror_length,
    mirror_thickness,
    source_length,
):
    """
    Convert placed nodes into Plotly-ready trace payloads.

    This helper does not add traces to `fig`; it only prepares structured data
    consumed by `_add_component_traces`.

    Extension points:
    - Add a new component symbol by extending `marker_traces`/`line_traces` or
      by introducing another geometry bucket (as done for `mirror_boxes`).
    - If you add component-level hover areas, include them in the returned
      component hover arrays so hover priority remains predictable.
    """
    marker_traces = {
        "beamsplitter": {"x": [], "y": [], "text": []},
        "directional_beamsplitter": {"x": [], "y": [], "text": []},
        "detector": {"x": [], "y": [], "text": [], "angle": []},
        "nothing": {"x": [], "y": [], "text": []},
        "unknown": {"x": [], "y": [], "text": []},
    }
    line_traces = {
        "laser": {"x": [], "y": [], "text": []},
        "squeezer": {"x": [], "y": [], "text": []},
    }
    mirror_boxes = []
    source_hover_traces = {
        "laser": {"x": [], "y": [], "text": []},
        "squeezer": {"x": [], "y": [], "text": []},
    }
    left_dots = {
        "mirror": {"x": [], "y": []},
        "beamsplitter": {"x": [], "y": []},
        "directional_beamsplitter": {"x": [], "y": []},
    }
    component_hover_x, component_hover_y = [], []
    component_hover_text, component_hover_sizes = [], []
    detector_angle_map = {"right": 0, "top": 90, "left": 180, "bottom": -90}

    for node, data in setup.nodes(data=True):
        if node not in positions or node in nodes_to_skip:
            continue
        component = data.get("component", "unknown")
        if component in _DETECTOR_LIKE_COMPONENTS:
            component = "detector"

        pos = positions[node]
        left_dir = left_dirs.get(node, "left")
        hover = _node_hover_text(node, data, component)

        if component == "beamsplitter":
            marker_traces["beamsplitter"]["x"].append(pos[0])
            marker_traces["beamsplitter"]["y"].append(pos[1])
            marker_traces["beamsplitter"]["text"].append(hover)
        elif component == "directional_beamsplitter":
            marker_traces["directional_beamsplitter"]["x"].append(pos[0])
            marker_traces["directional_beamsplitter"]["y"].append(pos[1])
            marker_traces["directional_beamsplitter"]["text"].append(hover)
        elif component == "detector":
            beam_dir = _beam_direction_for(node, adjacency, left_dirs)
            marker_traces["detector"]["x"].append(pos[0])
            marker_traces["detector"]["y"].append(pos[1])
            marker_traces["detector"]["text"].append(hover)
            marker_traces["detector"]["angle"].append(detector_angle_map.get(beam_dir, 0))
        elif component == "mirror":
            beam_dir = _port_direction(left_dir, "right")
            half_x, half_y = _mirror_half_extents(beam_dir, mirror_length, mirror_thickness)

            x0, x1 = pos[0] - half_x, pos[0] + half_x
            y0, y1 = pos[1] - half_y, pos[1] + half_y
            mirror_boxes.append(
                {
                    "x": [x0, x1, x1, x0, x0],
                    "y": [y0, y0, y1, y1, y0],
                    "text": hover,
                }
            )
        elif component in {"laser", "squeezer"}:
            beam_dir = _beam_direction_for(node, adjacency, left_dirs)
            if beam_dir in {"left", "right"}:
                p0 = (pos[0] - source_length / 2, pos[1])
                p1 = (pos[0] + source_length / 2, pos[1])
            else:
                p0 = (pos[0], pos[1] - source_length / 2)
                p1 = (pos[0], pos[1] + source_length / 2)
            _add_segment(line_traces[component], p0, p1, hover)
            # Dense invisible hover markers provide reliable full-length hover
            # coverage for source symbols in Plotly.
            p0_arr = np.array(p0, dtype=float)
            p1_arr = np.array(p1, dtype=float)
            n_hover_points = max(14, int(np.ceil(source_length / max(min_space, 1e-9) * 28)))
            t = np.linspace(0.0, 1.0, n_hover_points)
            hover_points = p0_arr[None, :] + (p1_arr - p0_arr)[None, :] * t[:, None]
            source_hover_traces[component]["x"].extend(hover_points[:, 0].tolist())
            source_hover_traces[component]["y"].extend(hover_points[:, 1].tolist())
            source_hover_traces[component]["text"].extend([hover] * n_hover_points)
        elif component == "nothing":
            marker_traces["nothing"]["x"].append(pos[0])
            marker_traces["nothing"]["y"].append(pos[1])
            marker_traces["nothing"]["text"].append(hover)
        else:
            marker_traces["unknown"]["x"].append(pos[0])
            marker_traces["unknown"]["y"].append(pos[1])
            marker_traces["unknown"]["text"].append(hover)

        component_hover_x.append(pos[0])
        component_hover_y.append(pos[1])
        component_hover_text.append(hover)
        component_hover_sizes.append(_component_hover_size(component))

        if component in {"mirror", "beamsplitter", "directional_beamsplitter"}:
            indicator_dir = _DIRECTION_VECTORS[left_dir]
            if component == "mirror":
                beam_dir = _port_direction(left_dir, "right")
                half_x, half_y = _mirror_half_extents(beam_dir, mirror_length, mirror_thickness)
                boundary_distance = half_x if left_dir in {"left", "right"} else half_y
                indicator_length = boundary_distance + 0.01 * min_space
            else:
                indicator_length = 0.14 * min_space
            center = pos + indicator_dir * indicator_length
            left_dots[component]["x"].append(center[0])
            left_dots[component]["y"].append(center[1])

    return (
        marker_traces,
        line_traces,
        mirror_boxes,
        source_hover_traces,
        left_dots,
        component_hover_x,
        component_hover_y,
        component_hover_text,
        component_hover_sizes,
    )


def _add_component_traces(
    fig,
    marker_traces,
    line_traces,
    mirror_boxes,
    source_hover_traces,
    left_dots,
    component_hover_x,
    component_hover_y,
    component_hover_text,
    component_hover_sizes,
    colors,
):
    """
    Append all component traces (visible + helper hover/dot traces) to `fig`.

    The function expects precomputed geometry from `_collect_component_trace_data`.
    Keep `legendgroup` synchronized for visible and helper traces to ensure grouped
    legend toggling works consistently.
    """
    line_specs = {
        "laser": {"width": 8, "name": "laser"},
        "squeezer": {"width": 8, "name": "squeezer"},
    }
    for component, spec in line_specs.items():
        trace = line_traces[component]
        if not trace["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode="lines",
                line=dict(color=colors[component], width=spec["width"]),
                name=spec["name"],
                legendgroup=component,
                text=trace["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for i, box in enumerate(mirror_boxes):
        fig.add_trace(
            go.Scatter(
                x=box["x"],
                y=box["y"],
                mode="lines",
                fill="toself",
                fillcolor=colors["mirror"],
                line=dict(color="black", width=1),
                name="mirror",
                legendgroup="mirror",
                showlegend=(i == 0),
                text=[box["text"]] * len(box["x"]),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for component in ("laser", "squeezer"):
        trace = source_hover_traces[component]
        if not trace["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode="markers",
                marker=dict(
                    size=30,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0),
                ),
                legendgroup=component,
                showlegend=False,
                text=trace["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    marker_specs = {
        "beamsplitter": {
            "symbol": "square",
            "size": 16,
            "name": "beamsplitter",
            "color_key": "beamsplitter",
        },
        "directional_beamsplitter": {
            "symbol": "diamond",
            "size": 16,
            "name": "directional bs",
            "color_key": "directional_beamsplitter",
        },
        "detector": {
            "symbol": "triangle-right",
            "size": 18,
            "name": "detector",
            "color_key": "detector",
        },
        "nothing": {
            "symbol": "diamond-open",
            "size": 14,
            "name": "nothing",
            "color_key": "nothing",
        },
        "unknown": {
            "symbol": "circle",
            "size": 12,
            "name": "unknown",
            "color_key": "unknown",
        },
    }
    for component, spec in marker_specs.items():
        trace = marker_traces[component]
        if not trace["x"]:
            continue
        marker_style = dict(
            symbol=spec["symbol"],
            size=spec["size"],
            color=colors[spec["color_key"]],
            line=dict(color="black", width=1),
        )
        if component == "nothing":
            marker_style["line"] = dict(color=colors[spec["color_key"]], width=1)
        if component == "detector":
            marker_style["angle"] = trace["angle"]

        fig.add_trace(
            go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode="markers",
                marker=marker_style,
                name=spec["name"],
                legendgroup=component,
                text=trace["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for component in ("mirror", "beamsplitter", "directional_beamsplitter"):
        dots = left_dots.get(component, {})
        if not dots.get("x"):
            continue
        fig.add_trace(
            go.Scatter(
                x=dots["x"],
                y=dots["y"],
                mode="markers",
                marker=dict(color="black", size=6),
                legendgroup=component,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if component_hover_x:
        fig.add_trace(
            go.Scatter(
                x=component_hover_x,
                y=component_hover_y,
                mode="markers",
                marker=dict(size=component_hover_sizes, color="rgba(0,0,0,0)", line=dict(width=0)),
                showlegend=False,
                text=component_hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )


def _beam_trace_updates(
    beam_records,
    source_name,
    frame_idx,
    source_config,
    global_max_by_source,
    min_beam_width,
    width_scale,
    max_beam_width,
):
    frame_count = source_config.frame_count_by_source[source_name]
    frame_idx = int(np.clip(frame_idx, 0, frame_count - 1))

    line_widths = []
    hover_widths = []
    line_texts = []
    hover_texts = []
    for record in beam_records:
        beam_width = _beam_width_for_record(
            record,
            source_name,
            frame_idx,
            global_max_by_source,
            min_beam_width,
            width_scale,
            max_beam_width,
        )
        beam_hover = _beam_hover_text(record, source_name, frame_idx, source_config)
        line_widths.append(beam_width)
        hover_widths.append(0.0)
        line_texts.append([beam_hover, beam_hover])
        hover_texts.append([beam_hover] * int(record.hover_point_count))

    return {"line.width": line_widths + hover_widths, "text": line_texts + hover_texts}


def _build_controls(
    beam_target_trace_indices,
    beam_records,
    source_config,
    global_max_by_source,
    min_beam_width,
    width_scale,
    max_beam_width,
):
    """
    Build Plotly UI controls for beam-width views.

    Returns:
    - `sliders`: optional signal-frequency slider configuration
    - `updatemenus`: optional powers/signals toggle buttons
    - `top_margin`: layout top margin required for visible controls
    - `legend_y`: y-position of the legend row

    Extension point:
    - Add new view modes by extending `beam_updates(...)` usage and adding
      corresponding buttons/slider entries here.
    """
    sliders = []
    updatemenus = []
    row_gap = 0.08
    base_row_y = 1.03
    legend_y = 1.02
    top_margin = 80

    def beam_updates(source_name, frame_idx):
        return _beam_trace_updates(
            beam_records,
            source_name,
            frame_idx,
            source_config,
            global_max_by_source,
            min_beam_width,
            width_scale,
            max_beam_width,
        )

    has_signal_slider = (
        "signals" in source_config.source_order
        and source_config.frame_count_by_source["signals"] > 1
        and len(beam_target_trace_indices) > 0
    )

    if has_signal_slider:
        slider_steps = []
        for frame_idx in range(source_config.frame_count_by_source["signals"]):
            if source_config.freq_values is not None:
                label = f"{float(source_config.freq_values[frame_idx]):.4g}"
            else:
                label = str(frame_idx)
            slider_steps.append(
                dict(
                    method="restyle",
                    args=[
                        beam_updates("signals", frame_idx),
                        beam_target_trace_indices,
                    ],
                    label=label,
                )
            )

        current_prefix = "frequency: " if source_config.freq_values is not None else "signal index: "
        sliders = [
            dict(
                active=source_config.initial_frame_by_source["signals"],
                currentvalue={"prefix": current_prefix},
                x=0.5,
                xanchor="center",
                y=base_row_y,
                yanchor="bottom",
                len=0.55,
                pad={"t": 8, "b": 0},
                steps=slider_steps,
                visible=(source_config.initial_source == "signals"),
            )
        ]

    has_mode_buttons = (
        "powers" in source_config.source_order
        and "signals" in source_config.source_order
        and len(beam_target_trace_indices) > 0
    )
    if has_mode_buttons:
        buttons_y = base_row_y + (row_gap if has_signal_slider else 0.0)
        buttons = [
            dict(
                label="powers",
                method="update",
                args=[
                    beam_updates("powers", 0),
                    {"sliders[0].visible": False} if has_signal_slider else {},
                    beam_target_trace_indices,
                ],
            ),
            dict(
                label="signals",
                method="update",
                args=[
                    beam_updates("signals", source_config.initial_frame_by_source["signals"]),
                    {"sliders[0].visible": True} if has_signal_slider else {},
                    beam_target_trace_indices,
                ],
            ),
        ]
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=buttons_y,
                yanchor="bottom",
                showactive=True,
                buttons=buttons,
            )
        ]

    if has_mode_buttons and has_signal_slider:
        legend_y = base_row_y + 2.0 * row_gap
    elif has_mode_buttons or has_signal_slider:
        legend_y = base_row_y + row_gap
    else:
        legend_y = 1.02

    row_count = 1 + int(has_mode_buttons) + int(has_signal_slider)
    top_margin = 80 + 30 * (row_count - 1)

    return sliders, updatemenus, top_margin, legend_y


def _resolve_html_output_path(output_file):
    out_path = os.fspath(output_file)
    if not str(out_path).lower().endswith(".html"):
        out_path = f"{out_path}.html"
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    return out_path


def _render_empty_setup(out_path):
    fig = go.Figure()
    fig.add_annotation(text="Empty setup", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return fig


def visualize_setup(
    setup,
    output_file="setup_visualization.html",
    *,
    seed=None,
    min_space=1.0,
    length_scale=0.001,
    attachment_length=None,
    port_to_index=None,
    powers=None,
    signals=None,
    frequencies=None,
    signal_index=0,
    min_beam_width=2.0,
    power_width_scale=5.0,
    max_beam_width=None,
    show_labels=True,
    dpi=200,
):
    """
    Visualize an interferometer setup as an interactive Plotly HTML diagram.

    Component names and properties are shown on hover only. Beam width can be
    scaled by `powers` and/or `signals` (switchable via in-HTML buttons).

    `output_file` is the output HTML filepath.
    """
    del show_labels, dpi  # Retained for API compatibility.
    out_path = _resolve_html_output_path(output_file)
    if attachment_length is None:
        attachment_length = 0.6 * min_space

    rng = np.random.default_rng(seed)
    node_components = {name: data.get("component", "") for name, data in setup.nodes(data=True)}
    nodes_to_skip = _collect_nodes_to_skip(setup, node_components)
    adjacency, edge_attached_nodes = _build_adjacency_and_attachments(
        setup,
        node_components,
        nodes_to_skip,
        min_space,
        length_scale,
        attachment_length,
    )
    positions, left_dirs, offset_x, component_spacing = _place_connected_nodes(
        setup,
        adjacency,
        node_components,
        nodes_to_skip,
        edge_attached_nodes,
        rng,
        min_space,
    )
    _place_edge_attached_nodes(
        edge_attached_nodes,
        positions,
        left_dirs,
        min_space,
        component_spacing,
        offset_x,
    )
    _reposition_detectors(
        setup,
        positions,
        left_dirs,
        node_components,
        nodes_to_skip,
        attachment_length,
    )

    if not positions:
        return _render_empty_setup(out_path)

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    margin = 1.5 * min_space
    x_range = [min(xs) - margin, max(xs) + margin]
    y_range = [min(ys) - margin, max(ys) + margin]

    mirror_length = 0.65 * min_space
    mirror_thickness = 0.12 * min_space
    source_length = 0.6 * min_space
    colors = {
        "beamsplitter": "#1f77b4",
        "directional_beamsplitter": "#ff7f0e",
        "mirror": "#999999",
        "laser": "#d62728",
        "squeezer": "#9467bd",
        "detector": "#8c564b",
        "nothing": "#7f7f7f",
        "unknown": "#17becf",
    }

    source_config = _prepare_beam_source_config(powers, signals, frequencies, signal_index)
    beam_records = _build_beam_records(
        setup,
        positions,
        node_components,
        port_to_index,
        source_config,
    )
    global_max_by_source = _global_max_series_by_source(beam_records, source_config)

    fig = go.Figure()
    beam_line_trace_indices, beam_hover_trace_indices = _add_beam_traces(
        fig,
        beam_records,
        source_config,
        global_max_by_source,
        min_space,
        source_length,
        min_beam_width,
        power_width_scale,
        max_beam_width,
    )
    _add_attachment_trace(fig, setup, positions, nodes_to_skip)

    (
        marker_traces,
        line_traces,
        mirror_boxes,
        source_hover_traces,
        left_dots,
        component_hover_x,
        component_hover_y,
        component_hover_text,
        component_hover_sizes,
    ) = _collect_component_trace_data(
        setup,
        positions,
        left_dirs,
        adjacency,
        nodes_to_skip,
        min_space,
        mirror_length,
        mirror_thickness,
        source_length,
    )
    _add_component_traces(
        fig,
        marker_traces,
        line_traces,
        mirror_boxes,
        source_hover_traces,
        left_dots,
        component_hover_x,
        component_hover_y,
        component_hover_text,
        component_hover_sizes,
        colors,
    )

    beam_target_trace_indices = beam_line_trace_indices + beam_hover_trace_indices
    sliders, updatemenus, top_margin, legend_y = _build_controls(
        beam_target_trace_indices,
        beam_records,
        source_config,
        global_max_by_source,
        min_beam_width,
        power_width_scale,
        max_beam_width,
    )

    fig.update_layout(
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            x=0.5,
            y=legend_y,
            xanchor="center",
            yanchor="bottom",
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        margin=dict(l=20, r=20, t=top_margin, b=20),
        showlegend=True,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    return fig
