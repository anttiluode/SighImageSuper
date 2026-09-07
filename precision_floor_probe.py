"""Precision-floor diagnostic for the Sigh recursive loop.

The old CUDA path stored every generation as float16. That changes a linear
recursion Ax into Q16(Ax), where Q16 is half-precision rounding.

This script uses an intentionally trivial operator A = 0.5 I. Such an operator
cannot change spectral shape at all in exact arithmetic: every Fourier component
must simply halve on every step. Therefore any late-emerging Fourier complexity
in the float16 path is unambiguously a quantization artifact.

The initial field contains exactly five real cosine modes (ten conjugate Fourier
bins). We compare float16 storage against float32 and float64 storage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


N = 64
DECAY = 0.5
STEPS = 30
RELATIVE_POWER_THRESHOLD = 1e-3


def initial_field() -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    modes = [(29, 30), (30, 31), (31, 29), (28, 31), (31, 28)]
    amplitudes = [1.0, 0.8, 0.6, 0.5, 0.4]

    x = np.zeros((N, N), dtype=np.float64)
    for (kx, ky), amp in zip(modes, amplitudes):
        phase = 0.3 * amp
        x += amp * np.cos(2.0 * np.pi * (kx * ii + ky * jj) / N + phase)

    x /= np.max(np.abs(x))
    return x


def spectral_support(x: np.ndarray) -> int:
    power = np.abs(np.fft.fft2(np.asarray(x, dtype=np.float64))) ** 2
    mx = float(power.max())
    if mx == 0.0:
        return 0
    return int(np.sum(power > mx * RELATIVE_POWER_THRESHOLD))


def metrics(x: np.ndarray) -> dict:
    xf = np.asarray(x, dtype=np.float64)
    nonzero = np.abs(xf[np.nonzero(xf)])
    return {
        "max_abs": float(np.max(np.abs(xf))),
        "std": float(np.std(xf)),
        "nonzero_values": int(np.count_nonzero(xf)),
        "min_positive_abs": float(nonzero.min()) if nonzero.size else 0.0,
        "spectral_bins_above_relative_threshold": spectral_support(xf),
        "quantized_levels": int(np.unique(x).size),
    }


def quantize(x: np.ndarray, storage: str) -> np.ndarray:
    if storage == "float16":
        return np.asarray(x, dtype=np.float16)
    if storage == "float32":
        return np.asarray(x, dtype=np.float32)
    if storage == "float64":
        return np.asarray(x, dtype=np.float64)
    raise ValueError(storage)


def run(storage: str) -> list[dict]:
    x = quantize(initial_field(), storage)
    rows = []
    checkpoints = {0, 18, 20, 21, 22, 23, 24, 25, 26, 30}

    for step in range(STEPS + 1):
        if step in checkpoints:
            row = {"step": step}
            row.update(metrics(x))
            rows.append(row)

        # A = 0.5 I. In exact arithmetic this NEVER changes Fourier shape.
        x = quantize(DECAY * np.asarray(x, dtype=np.float64), storage)

    return rows


def row_at(rows: list[dict], step: int) -> dict:
    return next(row for row in rows if row["step"] == step)


def main() -> None:
    f16 = run("float16")
    f32 = run("float32")
    f64 = run("float64")

    min_subnormal = float(np.nextafter(np.float16(0), np.float16(1)))

    result = {
        "gate": "precision_floor_diagnostic",
        "grid": [N, N],
        "operator": "A = 0.5 I (spectral shape must remain unchanged in exact arithmetic)",
        "initial_real_cosine_modes": 5,
        "ideal_fourier_support_bins": 10,
        "float16_min_positive_subnormal": min_subnormal,
        "relative_power_threshold": RELATIVE_POWER_THRESHOLD,
        "float16": f16,
        "float32": f32,
        "float64": f64,
        "key_receipt": {
            "float16_step_23_max_abs": row_at(f16, 23)["max_abs"],
            "float16_step_24_max_abs": row_at(f16, 24)["max_abs"],
            "float16_step_24_spectral_support": row_at(f16, 24)[
                "spectral_bins_above_relative_threshold"
            ],
            "float64_step_24_spectral_support": row_at(f64, 24)[
                "spectral_bins_above_relative_threshold"
            ],
            "float16_step_25_nonzero_values": row_at(f16, 25)["nonzero_values"],
        },
        "interpretation": (
            "With A=0.5I, exact arithmetic can only scale the field, so its ten-bin "
            "Fourier support cannot broaden. Float64 and float32 preserve that "
            "support while the amplitude keeps falling. Float16 storage reaches the "
            "5.9604645e-8 subnormal quantum; rounding turns the smooth five-mode field "
            "into sparse quantized patches and explodes the relative Fourier support "
            "before the state vanishes. The late 'last gasp' is therefore a numerical "
            "Q16 nonlinearity, not recovered physical/image information."
        ),
    }

    assert abs(min_subnormal - 2.0 ** -24) < 1e-20
    assert row_at(f64, 24)["spectral_bins_above_relative_threshold"] == 10
    assert row_at(f32, 24)["spectral_bins_above_relative_threshold"] == 10
    assert row_at(f16, 24)["max_abs"] == min_subnormal
    assert row_at(f16, 24)["spectral_bins_above_relative_threshold"] > 1000
    assert row_at(f16, 25)["nonzero_values"] == 0

    out = Path("results") / "precision_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
