import json
import logging
from collections import defaultdict
from pathlib import Path

from app.utils.appdata import AppData
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


def get_mapping_file(grid_size):
    """Return the mapping JSON file path for a supported grid size."""
    try:
        n = int(str(grid_size).split('x')[0])
    except Exception:
        raise ValueError(f"Invalid grid size format: {grid_size}")

    if n not in SUPPORTED_GRID_SIZES:
        raise ValueError(f"Unsupported grid size: {grid_size}. Supported sizes: {sorted(SUPPORTED_GRID_SIZES)}")

    mapping_file = Path(__file__).parent / f"{n}_mode_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found for {grid_size}: {mapping_file}")
    return mapping_file


def import_mapping_json(json_str):
    """Imports and validates JSON mapping, returns label map."""
    try:
        data = json.loads(json_str)
        validate(instance=data, schema=MAPPING_SCHEMA)

        label_map = {}
        for label, values in data.items():
            label_map[label] = (values["theta"], values["phi"])
        return label_map
    except Exception as e:
        raise ValueError(f"Invalid mapping JSON: {str(e)}") from e


def load_custom_label_mapping(json_path):
    """Loads a manually defined label-to-channel mapping from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            json_str = f.read()
        return import_mapping_json(json_str)
    except Exception as e:
        raise ValueError(f"Failed to load label mapping: {e}") from e


def create_label_mapping(grid_n):
    """Load the label-to-channel mapping from the corresponding JSON file."""
    mapping_file = Path(__file__).parent / f"{grid_n}_mode_mapping.json"
    try:
        with open(mapping_file, 'r') as f:
            json_str = f.read()
        return import_mapping_json(json_str)
    except Exception as e:
        raise ValueError(f"Failed to load label mapping for {grid_n}x{grid_n}: {e}") from e


def print_mapping(label_map):
    """Prints mapping in column groups with channel pairs."""
    columns = defaultdict(list)
    for label, chs in label_map.items():
        col = label[0]
        columns[col].append((label, chs))

    for col in sorted(columns.keys()):
        logging.info(f"Column {col}:")
        for label, (theta, phi) in sorted(columns[col], key=lambda x: int(x[0][1:])):
            logging.info(f"  {label}: θ{theta}, φ{phi}")


def clamp_value(value, max_limit):
    """Safely clamp input values."""
    try:
        val = float(value)
        return max(min(val, max_limit), 0.0)
    except (ValueError, TypeError):
        return 0.0


def apply_qontrol_mapping(qontrol_device, channel_map):
    """Apply mapped values to physical device."""
    if not qontrol_device.device:
        logging.info("Qontrol device not connected")
        return

    for channel, current in channel_map.items():
        try:
            qontrol_device.set_current(channel, current)
        except Exception as e:
            logging.error(f"Channel {channel} error: {str(e)}")


def apply_grid_mapping(qontrol_device, grid_data, grid_size):
    """Map grid values to Qontrol channels and write them to the device."""
    try:
        label_map = create_label_mapping(int(str(grid_size).split('x')[0]))

        if isinstance(grid_data, str):
            export_data = json.loads(grid_data)
        else:
            export_data = grid_data

        channel_values = {}
        current_limit = qontrol_device.globalcurrentlimit

        for label, data in export_data.items():
            if label in label_map:
                theta_ch, phi_ch = label_map[label]
                theta = clamp_value(data.get("theta", 0), current_limit)
                phi = clamp_value(data.get("phi", 0), current_limit)
                channel_values[theta_ch] = theta
                channel_values[phi_ch] = phi

        apply_qontrol_mapping(qontrol_device, channel_values)
    except Exception as e:
        logging.error(f"Mapping error: {str(e)}")


def get_mapping_functions(grid_size=None):
    """Return create_label_mapping and apply_grid_mapping for the given grid size."""
    if grid_size is None:
        grid_size = getattr(AppData, 'grid_size', '8x8')

    if int(str(grid_size).split('x')[0]) not in SUPPORTED_GRID_SIZES:
        raise ValueError(f"Unsupported grid size: {grid_size}")

    return create_label_mapping, apply_grid_mapping
