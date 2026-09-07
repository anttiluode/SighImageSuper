"""Gate 2: the interesting part is the transient before the checkerboard.

For the current Sigh operator the Fourier basis diagonalizes the map, so the
singular values of A^t are simply |gain_k|^t.  This lets us measure, without
sampling any particular image, how many independent directions remain large
enough to matter as recursive depth increases.

This is NOT a claim about task memory capacity.  It is an operator diagnostic:
how quickly a 4096-dimensional state map becomes effectively low-dimensional.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate1_nyquist_envelope import N, sigh_high_pass_operator


STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
THRESHOLDS = (1e-3, 1e-6)


def run() -> dict:
    gain = np.abs(sigh_high_pass_operator(N)).ravel()
    rows = []

    for step in STEPS:
        singular = gain ** step
        s2 = singular * singular
        s4 = s2 * s2
        max_s2 = float(s2.max())

        stable_rank = float(s2.sum() / max_s2)
        effective_singular_modes = float(
            (s2.sum() ** 2) / (s4.sum() + 1e-300)
        )

        row = {
            "step": step,
            "stable_rank": stable_rank,
            "effective_singular_modes": effective_singular_modes,
        }
        for threshold in THRESHOLDS:
            row[f"directions_above_{threshold:g}"] = int(
                np.count_nonzero(singular > threshold)
            )
        rows.append(row)

    return {
        "grid": [N, N],
        "state_dimension": N * N,
        "max_gain": float(gain.max()),
        "min_gain": float(gain.min()),
        "steps": rows,
        "interpretation": (
            "The algebraic map remains full-rank because every filter gain is "
            "positive, but its usable/transient dimensionality collapses rapidly. "
            "The pure checkerboard is the one-dimensional endpoint; the richer "
            "cloud is the transient regime in which many operator directions are "
            "still distinguishable above finite precision or noise."
        ),
    }


def main() -> None:
    receipt = run()
    out = Path("results/gate2_transient_dimension.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print("Gate 2 — transient operator dimension")
    for row in receipt["steps"]:
        print(
            f"step {row['step']:3d}  "
            f"stable_rank={row['stable_rank']:.6g}  "
            f"effective={row['effective_singular_modes']:.6g}  "
            f">1e-3={row['directions_above_0.001']:4d}  "
            f">1e-6={row['directions_above_1e-06']:4d}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
