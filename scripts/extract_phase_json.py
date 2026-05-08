#!/usr/bin/env python3
"""Export phase settings from a .npz archive into JSON for import_paths_json()."""

import argparse
import json
import os
import sys

import numpy as np

# Ensure the repository root is on sys.path when running this script directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.engine import Engine


def load_phase_archive(npz_path):
    with np.load(npz_path) as data:
        if 'thetas' not in data or 'phis' not in data:
            raise ValueError(f"NPZ archive must contain 'thetas' and 'phis' arrays. Found: {list(data.keys())}")
        return data['thetas'], data['phis']


def build_import_json(thetas, phis, arms=None, n_modes=8):
    if arms is None:
        arms = ['TL', 'TR', 'BL', 'BR']

    engine = Engine(n_modes=n_modes)
    if len(thetas) != engine.n_mzis or len(phis) != engine.n_mzis:
        raise ValueError(
            f"Expected {engine.n_mzis} phase values for this {n_modes}-mode engine, got "
            f"{len(thetas)} thetas and {len(phis)} phis"
        )

    data = {}
    for mid, theta, phi in zip(engine.mzi_ids, thetas, phis):
        theta_pi = float(theta) / np.pi
        phi_pi = float(phi) / np.pi
        data[mid] = {
            'arms': arms,
            'theta': str(theta_pi),
            'phi': str(phi_pi),
        }
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract theta/phi values from a unitary NPZ file and export JSON compatible with import_paths_json().'
    )
    parser.add_argument(
        'input_npz',
        nargs='?',
        default='unitary/retrieved_unitary.npz',
        help='Path to the input .npz archive containing thetas and phis arrays.'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Path to the output JSON file. Defaults to the same base name as the input file with .json extension.'
    )
    parser.add_argument(
        '--n-modes',
        type=int,
        default=8,
        help='Number of spatial modes used by the engine layout (default: 8).'
    )
    parser.add_argument(
        '--arms',
        nargs='+',
        default=['TL', 'TR', 'BL', 'BR'],
        help='List of arms to include for each center. Defaults to all four arms.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.input_npz)
        output_path = f"{base}.json"

    thetas, phis = load_phase_archive(args.input_npz)
    output_data = build_import_json(thetas, phis, arms=args.arms, n_modes=args.n_modes)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(output_data, out_file, indent=2)

    print(f"Exported {len(output_data)} centers to {output_path}")


if __name__ == '__main__':
    main()
