#!/usr/bin/env python3
"""
Gate 0: analytical receipt for SighImageSuper's 64x64 recursive high-pass loop.

The live app applies a diagonal Fourier operator whose gain is a function of
normalized squared radial frequency k^2.  The high-pass preset reaches gain 1.0
only at the joint Nyquist bin of a 64x64 grid.  This script predicts the
asymptotic surviving subspace directly from the filter, then verifies it by
iteration.

No GUI and no torch are required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HIGH_PASS_GAINS = np.array(
    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    dtype=np.float64,
)


def build_filter(n: int = 64, num_points: int = 256) -> np.ndarray:
    # Match torch.fft.fftfreq(n, d=1/n): integer-valued DFT frequencies.
    k = np.fft.fftfreq(n, d=1 / n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    k2 = k2 / k2.max()

    band_centers = np.linspace(0.0, 1.0, len(HIGH_PASS_GAINS))
    x_coords = np.linspace(0.0, 1.0, num_points)
    lookup = np.interp(x_coords, band_centers, HIGH_PASS_GAINS)

    indices = np.floor(k2 * (num_points - 1)).astype(np.int64)
    indices = np.clip(indices, 0, num_points - 1)
    return lookup[indices]


def synthetic_image(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Smooth-ish structured field plus tiny broadband component so every mode
    # has a reproducible nonzero chance of being seeded.
    y, x = np.mgrid[0:n, 0:n]
    img = (
        0.7 * np.exp(-((x - 0.32 * n) ** 2 + (y - 0.45 * n) ** 2) / (0.08 * n * n))
        + 0.5 * np.exp(-((x - 0.72 * n) ** 2 + (y - 0.62 * n) ** 2) / (0.04 * n * n))
        + 0.03 * rng.standard_normal((n, n))
    )
    img -= img.min()
    img /= img.max() + 1e-30
    return img


def load_image(path: str, n: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required when --image is used.") from exc

    img = Image.open(path).convert("L").resize((n, n))
    arr = np.asarray(img, dtype=np.float64)
    arr -= arr.min()
    arr /= arr.max() + 1e-30
    return arr


def iterate(x0: np.ndarray, filt: np.ndarray, steps: int) -> np.ndarray:
    x = x0.copy()
    for _ in range(steps):
        x = np.fft.ifft2(np.fft.fft2(x) * filt).real
    return x


def asymptotic_projection(x0: np.ndarray, filt: np.ndarray, atol: float = 1e-15):
    max_gain = float(np.max(np.abs(filt)))
    mask = np.isclose(np.abs(filt), max_gain, atol=atol, rtol=0.0)

    f0 = np.fft.fft2(x0)
    projected_fft = np.zeros_like(f0)
    projected_fft[mask] = f0[mask]
    projection = np.fft.ifft2(projected_fft).real
    return max_gain, mask, projection


def checkerboard(n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n]
    return ((-1.0) ** (x + y)).astype(np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel()
    bb = b.ravel()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.n % 2:
        raise SystemExit("Gate 0 uses an even grid so the Nyquist checkerboard exists exactly.")

    filt = build_filter(args.n)
    x0 = load_image(args.image, args.n) if args.image else synthetic_image(args.n, args.seed)

    max_gain, mask, predicted = asymptotic_projection(x0, filt)
    observed = iterate(x0, filt, args.steps)

    max_bins = np.argwhere(mask)
    next_gain = float(np.max(np.abs(filt[~mask]))) if np.any(~mask) else float("nan")

    residual = observed - predicted * (max_gain ** args.steps)
    rel_error = float(
        np.linalg.norm(residual)
        / (np.linalg.norm(observed) + np.linalg.norm(predicted) + 1e-30)
    )

    cb = checkerboard(args.n)
    pred_cb = abs(cosine_similarity(predicted, cb))
    obs_cb = abs(cosine_similarity(observed, cb))

    report = {
        "grid": [args.n, args.n],
        "steps": args.steps,
        "max_abs_gain": max_gain,
        "num_max_gain_bins": int(mask.sum()),
        "max_gain_bins_array_indices": max_bins.tolist(),
        "next_largest_abs_gain": next_gain,
        "predicted_projection_std": float(predicted.std()),
        "observed_std": float(observed.std()),
        "predicted_checkerboard_abs_cosine": pred_cb,
        "observed_checkerboard_abs_cosine": obs_cb,
        "relative_error_vs_analytical_limit": rel_error,
        "interpretation": (
            "The high-pass loop preserves exactly the joint Nyquist checkerboard mode."
            if max_gain == 1.0 and int(mask.sum()) == 1
            else "The loop converges toward the maximum-gain Fourier subspace."
        ),
    }

    print(json.dumps(report, indent=2))

    # Strong but numerically tolerant gates.
    assert math.isclose(max_gain, 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert int(mask.sum()) == 1
    assert max_bins.tolist() == [[args.n // 2, args.n // 2]]
    assert pred_cb > 0.999999
    assert obs_cb > 0.999
    assert rel_error < 1e-10

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
