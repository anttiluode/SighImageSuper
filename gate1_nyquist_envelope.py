"""Gate 1: expose the 'cloud' as a Nyquist-carrier envelope.

The live high-pass loop eventually leaves the joint Nyquist checkerboard mode.
Before that endpoint, states near the Nyquist corner can be written as a fast
checkerboard carrier multiplied by a slower spatial envelope.

Multiplying the state by (-1)^(i+j) demodulates that carrier back to DC.  In
Fourier space this is exactly a half-grid shift on each axis.  This gate checks
that identity and measures how the demodulated envelope loses non-DC modes
under the exact SighImageSuper high-pass operator.

No learning is involved here; this is an analytical/control diagnostic for the
visual cloud seen in the recursive GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N = 64
STEPS = (0, 1, 4, 16, 64, 256, 1024)
HIGH_PASS_GAINS = np.array(
    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    dtype=np.float64,
)


def sigh_high_pass_operator(n: int = N) -> np.ndarray:
    """Reproduce the GUI's 256-point k^2 lookup and high-pass preset."""
    lookup_x = np.linspace(0.0, 1.0, 256)
    band_x = np.linspace(0.0, 1.0, len(HIGH_PASS_GAINS))
    lookup = np.interp(lookup_x, band_x, HIGH_PASS_GAINS)

    freq = np.fft.fftfreq(n, d=1 / n)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx * kx + ky * ky
    k2 = k2 / k2.max()
    indices = np.floor(k2 * (len(lookup) - 1)).astype(np.int64)
    return lookup[indices]


def checkerboard(n: int = N) -> np.ndarray:
    ii, jj = np.indices((n, n))
    return np.where((ii + jj) % 2 == 0, 1.0, -1.0)


def spectral_metrics(x: np.ndarray) -> dict[str, float]:
    power = np.abs(np.fft.fft2(x)) ** 2
    total = float(power.sum()) + 1e-300
    p = power / total
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    effective_modes = float(1.0 / np.sum(p * p))
    dc_fraction = float(p[0, 0])
    return {
        "spectral_entropy": entropy,
        "effective_modes": effective_modes,
        "dc_power_fraction": dc_fraction,
    }


def evolve_exact(x0: np.ndarray, operator: np.ndarray, step: int) -> np.ndarray:
    spectrum = np.fft.fft2(x0)
    return np.fft.ifft2(spectrum * (operator ** step)).real


def run(seed: int = 13) -> dict:
    operator = sigh_high_pass_operator(N)
    carrier = checkerboard(N)

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(N, N))

    # Multiplication by the checkerboard is exact Nyquist demodulation:
    # FFT[checkerboard * x] = FFT[x] shifted by N/2 in both axes.
    x_fft = np.fft.fft2(x0)
    demod_fft = np.fft.fft2(carrier * x0)
    shifted = np.roll(np.roll(x_fft, N // 2, axis=0), N // 2, axis=1)
    demodulation_identity_max_error = float(np.max(np.abs(demod_fft - shifted)))

    max_gain = float(operator.max())
    max_bins = np.argwhere(np.isclose(operator, max_gain, rtol=0.0, atol=0.0))

    rows = []
    for step in STEPS:
        x = evolve_exact(x0, operator, step)
        envelope = carrier * x
        metrics = spectral_metrics(envelope)
        rows.append(
            {
                "step": int(step),
                "raw_std": float(x.std()),
                **metrics,
            }
        )

    final = rows[-1]
    return {
        "grid": [N, N],
        "seed": seed,
        "operator": "exact SighImageSuper high-pass preset",
        "max_gain": max_gain,
        "max_gain_bins_array_indices": max_bins.tolist(),
        "checkerboard_demodulation_identity_max_error": demodulation_identity_max_error,
        "demodulated_unit_gain_bin": [0, 0],
        "steps": rows,
        "final_dc_power_fraction": final["dc_power_fraction"],
        "final_effective_modes": final["effective_modes"],
        "interpretation": (
            "The pre-checkerboard cloud is a Nyquist-carrier envelope. "
            "Multiplying by (-1)^(i+j) shifts the Nyquist corner to DC exactly; "
            "recursive high-pass selection then removes the envelope's non-DC "
            "components until the envelope is constant, which remodulates to a "
            "pure checkerboard."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--out", type=Path, default=Path("results/gate1_envelope.json")
    )
    args = parser.parse_args()

    receipt = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print("Gate 1 — Nyquist envelope")
    print(
        "checkerboard demodulation identity max error:",
        f"{receipt['checkerboard_demodulation_identity_max_error']:.3e}",
    )
    for row in receipt["steps"]:
        print(
            f"step {row['step']:4d}  "
            f"std={row['raw_std']:.6g}  "
            f"DC={row['dc_power_fraction']:.9f}  "
            f"modes={row['effective_modes']:.6g}  "
            f"H={row['spectral_entropy']:.6g}"
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
