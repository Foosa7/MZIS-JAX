"""Extract physical heater/MMI coordinates from a chip GDS into layout/<n>mode_geometry.json.

Needs `gdstk` (not in requirements.txt -- `pip install gdstk`). Run from the repo root:

    python scripts/extract_layout_geometry.py --gds /path/to/8x8-chip.gds --modes 8

The mesh is laid out with propagation along +x and modes stacked along y at the
fibre-array pitch. Each MZI occupies two heater columns and two MMI columns,
interleaved as  phi-heater | MMI | theta-heater | MMI  along the propagation
direction, which is exactly the engine's BS(e_r).diag(e^{i.theta},1).BS(e_l)
convention with phi applied externally on the top input.

Both heaters of an MZI sit on the *upper* arm (higher y), which is the lower mode
index, so mode 0 is the top of the die and mode y decreases by the mode pitch.
That fixes the id assignment: within a column, MZI k+1 has mode_top = start+2k,
so its heaters are at the k-th row counted downward from the top.
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Blackbox cell names differ per foundry/design flow; both known chips are covered.
CELL_ALIASES = {
    "TO_PhaseShifter": "HTR",                         # 8-mode (Nazca, SOI250)
    "AMF_SOI_TOPhaseShifter2_0114c11b": "HTR",        # 12-mode (gdsfactory, AMF)
    "SOI250_MMI2x2_eh250_SiO2up": "MMI",
    "AMF_SOI_2X2MMI_Cband_v3p0": "MMI",
}


def _offsets(ref):
    """Instance offsets for a reference, handling gdstk's empty-Repetition case."""
    rep = ref.repetition
    if rep is None:
        return [(0.0, 0.0)]
    try:
        offs = [(float(a), float(b)) for a, b in rep.get_offsets()]
    except Exception:
        return [(0.0, 0.0)]
    return offs or [(0.0, 0.0)]


def _walk(cells, cell, ox, oy, rot, mag, xrefl, out, depth=0):
    """Recursively flatten the hierarchy, accumulating placed instances of CELL_ALIASES."""
    if depth > 30:
        return
    for ref in cell.references:
        name = ref.cell.name if hasattr(ref.cell, "name") else str(ref.cell)
        rad = math.radians(rot)
        ca, sa = math.cos(rad), math.sin(rad)
        px, py = ref.origin[0] * mag, ref.origin[1] * mag
        if xrefl:
            py = -py
        gx = ox + px * ca - py * sa
        gy = oy + px * sa + py * ca
        nrot = rot + ref.rotation * (-1 if xrefl else 1)
        nmag = mag * ref.magnification
        nxrefl = xrefl ^ ref.x_reflection
        for dx, dy in _offsets(ref):
            ddx, ddy = dx * mag, dy * mag
            fx = gx + ddx * ca - ddy * sa
            fy = gy + ddx * sa + ddy * ca
            if name in CELL_ALIASES:
                out.append((CELL_ALIASES[name], fx, fy))
            elif name in cells:
                _walk(cells, cells[name], fx, fy, nrot, nmag, nxrefl, out, depth + 1)


def _cluster(values, tol):
    """1-D single-linkage clustering; returns sorted cluster centres."""
    vals = sorted(values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _column_name(idx):
    """Same A, B, ... AA scheme as Engine._define_layout."""
    chars = []
    while idx >= 0:
        chars.append(chr(65 + (idx % 26)))
        idx = idx // 26 - 1
    return "".join(reversed(chars))


def extract(gds_path, n_modes, top_cell=None, tol=5.0):
    import gdstk

    lib = gdstk.read_gds(gds_path)
    cells = {c.name: c for c in lib.cells}

    if top_cell is not None:
        tops = [cells[top_cell]]
    else:
        tops = list(lib.top_level())

    n_mzis = (n_modes // 2) * (n_modes // 2) + ((n_modes - 1) // 2) * (n_modes - (n_modes // 2))
    # Clements on n modes has n columns alternating n//2 and (n-1)//2 MZIs.
    n_mzis = sum((n_modes // 2) if i % 2 == 0 else (n_modes - 1) // 2 for i in range(n_modes))

    chosen, placed = None, None
    for cell in tops:
        out = []
        _walk(cells, cell, 0.0, 0.0, 0.0, 1.0, False, out)
        counts = collections.Counter(t for t, _, _ in out)
        if counts.get("HTR", 0) == 2 * n_mzis:
            chosen, placed = cell.name, out
            break
    if placed is None:
        raise SystemExit(
            "no top cell has 2*%d = %d heaters; found %s"
            % (n_mzis, 2 * n_mzis, [
                (c.name, collections.Counter(
                    t for t, _, _ in
                    (lambda o: (_walk(cells, c, 0, 0, 0, 1.0, False, o), o)[1])([])
                ).get("HTR", 0)) for c in tops
            ])
        )

    heaters = [(x, y) for t, x, y in placed if t == "HTR"]
    mmis = [(x, y) for t, x, y in placed if t == "MMI"]

    # Device extents along propagation, needed to size the source line and the
    # victim arm segment. Cells are drawn with propagation along their local x.
    def _cell_len(alias):
        for name, tag in CELL_ALIASES.items():
            if tag == alias and name in cells:
                (x0, _), (x1, _) = cells[name].bounding_box()
                return x1 - x0
        return None

    heater_len = _cell_len("HTR")
    mmi_len = _cell_len("MMI")

    # The two dies use opposite orientations (8-mode propagates along x, 12-mode
    # along y). Pick the propagation axis as the one carrying 2*n_modes heater
    # columns, and transpose so the rest of this function is axis-agnostic.
    cx = _cluster([p[0] for p in heaters], tol)
    cy = _cluster([p[1] for p in heaters], tol)
    if len(cx) == 2 * n_modes:
        prop_axis, hx, hy = "x", cx, cy
    elif len(cy) == 2 * n_modes:
        prop_axis = "y"
        heaters = [(y, x) for x, y in heaters]
        mmis = [(y, x) for x, y in mmis]
        hx, hy = cy, cx
    else:
        raise SystemExit(
            "expected %d heater columns on one axis, got x=%d y=%d"
            % (2 * n_modes, len(cx), len(cy))
        )
    mmi_y = _cluster([p[1] for p in mmis], tol)

    mode_pitch = (hy[-1] - hy[0]) / (len(hy) - 1)

    def build(descending):
        """Assign ids assuming mode 0 is at the high (or low) end of the mode axis.

        The engine's theta acts on mode_top, and both heaters of an MZI sit on one
        arm, so exactly one of the two orientations places a heater at every
        predicted coordinate. Try both rather than assume.
        """
        if descending:
            mode_y = [hy[-1] - mode_pitch * m for m in range(n_modes)]
        else:
            mode_y = [hy[0] + mode_pitch * m for m in range(n_modes)]
        recs = {}
        for c in range(n_modes):
            col = _column_name(c)
            start_mode = 0 if c % 2 == 0 else 1
            count = n_modes // 2 if c % 2 == 0 else (n_modes - 1) // 2
            phi_x, theta_x = hx[2 * c], hx[2 * c + 1]
            for k in range(count):
                mode_top = start_mode + 2 * k
                y = mode_y[mode_top]
                for hxx in (phi_x, theta_x):
                    if not any(
                        abs(p[0] - hxx) <= tol and abs(p[1] - y) <= tol for p in heaters
                    ):
                        return None, None
                recs["%s%d" % (col, k + 1)] = {
                    "column": c,
                    "mode_top": mode_top,
                    "phi_xy": [round(phi_x, 3), round(y, 3)],
                    "theta_xy": [round(theta_x, 3), round(y, 3)],
                    "arm_y": [
                        round(mode_y[mode_top], 3),
                        round(mode_y[mode_top + 1], 3),
                    ],
                }
        return recs, mode_y

    records, mode_y = build(descending=True)
    mode0_at = "high"
    if records is None:
        records, mode_y = build(descending=False)
        mode0_at = "low"
    if records is None:
        raise SystemExit(
            "neither mode-axis orientation places a heater at every predicted "
            "MZI coordinate -- the heater is probably not on the mode_top arm"
        )

    return {
        "source_gds": os.path.basename(gds_path),
        "top_cell": chosen,
        "n_modes": n_modes,
        "units": "um",
        # Coordinates below are in a canonical frame: first component runs along
        # propagation, second along the mode stack. `prop_axis` records which die
        # axis that was, `mode0_at` which end of the mode axis holds mode 0.
        "prop_axis": prop_axis,
        "mode0_at": mode0_at,
        "n_heaters": len(heaters),
        "n_mmis": len(mmis),
        "mode_pitch": round(mode_pitch, 3),
        "arm_pitch": round(mode_pitch, 3),
        "intra_mzi_x": round(hx[1] - hx[0], 3),
        "column_pitch_x": round(hx[2] - hx[0], 3),
        "same_column_neighbour": round(2 * mode_pitch, 3),
        "heater_length": round(heater_len, 3) if heater_len else None,
        "mmi_length": round(mmi_len, 3) if mmi_len else None,
        # The waveguide stretch a heater's phase is lumped onto: between the two
        # MMIs bracketing it, which is centred on the heater itself.
        "segment_length": (
            round((hx[2] - hx[0]) / 2.0 - mmi_len, 3) if mmi_len else None
        ),
        "mode_y": [round(v, 3) for v in mode_y],
        "mmi_y_rows": [round(v, 3) for v in mmi_y],
        "mzis": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gds", required=True)
    ap.add_argument("--modes", type=int, required=True)
    ap.add_argument("--top-cell", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    geo = extract(args.gds, args.modes, args.top_cell)
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "layout",
        "%dmode_geometry.json" % args.modes,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(geo, fh, indent=2)

    print("top cell        : %s" % geo["top_cell"])
    print("heaters / MMIs  : %d / %d" % (geo["n_heaters"], geo["n_mmis"]))
    print("MZIs            : %d" % len(geo["mzis"]))
    print("mode pitch      : %.1f um" % geo["mode_pitch"])
    print("phi->theta (x)  : %.1f um" % geo["intra_mzi_x"])
    print("column pitch (x): %.1f um" % geo["column_pitch_x"])
    print("same-column nbr : %.1f um" % geo["same_column_neighbour"])
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
