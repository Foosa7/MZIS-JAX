"""Mesh label (A1, G4, ...) to driver channel mapping.

Previously this imported `app.utils.appdata`, a package that does not exist
in this repository, so the module could not be imported at all. The grid size
is now an explicit argument instead of ambient global state.

Writes go through `SafeDriver`, which owns the current limits. Errors are
raised rather than logged and swallowed: a mapping failure that returns
quietly would leave the caller believing the chip was programmed.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from jsonschema import validate

MAPPING_SCHEMA = {
    "type": "object",
    "patternProperties": {
        "^[A-Z][1-9][0-9]*$": {
            "type": "object",
            "properties": {
                "theta": {"type": "integer", "minimum": 0},
                "phi": {"type": "integer", "minimum": 0},
                "control_type": {"type": "string"}
            },
            "required": ["theta", "phi"],
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

SUPPORTED_GRID_SIZES = {8, 12}


def grid_n(grid_size):
    """Normalises '8x8', '8', or 8 into the integer 8."""
    try:
        n = int(str(grid_size).split('x')[0])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid grid size format: {grid_size!r}")
    if n not in SUPPORTED_GRID_SIZES:
        raise ValueError(
            f"Unsupported grid size: {grid_size}. "
            f"Supported: {sorted(SUPPORTED_GRID_SIZES)}"
        )
    return n


def get_mapping_file(grid_size):
    """Path of the mapping JSON for a supported grid size."""
    mapping_file = Path(__file__).parent / f"{grid_n(grid_size)}_mode_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found for {grid_size}: {mapping_file}")
    return mapping_file


def import_mapping_json(json_str):
    """Validates a mapping document and returns {label: (theta_ch, phi_ch)}."""
    try:
        data = json.loads(json_str)
        validate(instance=data, schema=MAPPING_SCHEMA)
    except Exception as e:
        raise ValueError(f"Invalid mapping JSON: {e}") from e

    label_map = {label: (v["theta"], v["phi"]) for label, v in data.items()}

    # A duplicated channel would make two heaters fight over one output.
    channels = [ch for pair in label_map.values() for ch in pair]
    duplicates = {ch for ch in channels if channels.count(ch) > 1}
    if duplicates:
        raise ValueError(f"Mapping assigns channel(s) {sorted(duplicates)} more than once")

    return label_map


def load_custom_label_mapping(json_path):
    """Loads a hand-written label-to-channel mapping."""
    try:
        with open(json_path, 'r') as f:
            return import_mapping_json(f.read())
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to load label mapping: {e}") from e


def create_label_mapping(grid_size):
    """Loads the shipped mapping for a grid size."""
    mapping_file = get_mapping_file(grid_size)
    with open(mapping_file, 'r') as f:
        return import_mapping_json(f.read())


def print_mapping(label_map):
    """Logs the mapping grouped by mesh column."""
    columns = defaultdict(list)
    for label, chs in label_map.items():
        columns[label[0]].append((label, chs))

    for col in sorted(columns):
        logging.info("Column %s:", col)
        for label, (theta, phi) in sorted(columns[col], key=lambda x: int(x[0][1:])):
            logging.info("  %s: theta=ch%s, phi=ch%s", label, theta, phi)


def apply_channel_currents(driver, channel_currents, user=None, job_id=None):
    """Writes a {channel: mA} map through the safety wrapper."""
    if not driver.connected:
        raise RuntimeError("current driver is not connected")
    return driver.set_many(channel_currents, user=user, job_id=job_id)


def grid_currents_to_channels(grid_currents, grid_size):
    """Turns {label: {'theta': mA, 'phi': mA}} into {channel: mA}.

    Values must already be currents. Anything non-numeric raises, because the
    failure mode this replaces was writing raw phase values to the DACs.
    """
    label_map = create_label_mapping(grid_size)
    if isinstance(grid_currents, str):
        grid_currents = json.loads(grid_currents)

    channel_values = {}
    for label, data in grid_currents.items():
        if label not in label_map:
            continue
        theta_ch, phi_ch = label_map[label]
        for arm, channel in (("theta", theta_ch), ("phi", phi_ch)):
            if arm not in data or data[arm] is None:
                continue
            try:
                channel_values[channel] = float(data[arm])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{label}.{arm} is {data[arm]!r}, which is not a current in mA"
                ) from e
    return channel_values


def apply_grid_mapping(driver, grid_currents, grid_size, user=None, job_id=None):
    """Maps a grid of currents onto channels and writes them to the device."""
    return apply_channel_currents(
        driver, grid_currents_to_channels(grid_currents, grid_size),
        user=user, job_id=job_id,
    )
