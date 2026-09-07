"""Gate 4: travelling trace with no persistent eigenmode.

This is the counterexample to an overly eigenvalue-centered memory story.

An 8-cell directed delay line receives one scalar input per step at cell 0.
The state update is

    x_(t+1) = S x_t + B u_t

where S shifts activity one cell downstream.  S is nilpotent: every eigenvalue
is exactly zero and S**8 = 0.  Nevertheless, after eight inputs the present
state contains all eight values.  A single sensor C on the final cell can read
the entire history out over the next eight silent steps.

So a system can carry temporal information by moving it through state space,
not by keeping any long-lived eigenmode alive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


N = 8
INPUTS = np.array([0.25, -0.8, 0.4, 1.1, -0.35, 0.6, -1.2, 0.9], dtype=float)


def shift_operator(n: int = N) -> np.ndarray:
    S = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        S[i + 1, i] = 1.0
    return S


def main() -> None:
    S = shift_operator()
    B = np.zeros(N, dtype=float)
    B[0] = 1.0
    C = np.zeros(N, dtype=float)
    C[-1] = 1.0

    x = np.zeros(N, dtype=float)
    for u in INPUTS:
        x = S @ x + B * float(u)

    loaded_state = x.copy()

    # Read through one bounded sensor while no new input is supplied.
    recovered = []
    for _ in range(N):
        recovered.append(float(C @ x))
        x = S @ x

    eig = np.linalg.eigvals(S)
    spectral_radius = float(np.max(np.abs(eig)))
    nilpotent_residual = float(np.linalg.norm(np.linalg.matrix_power(S, N)))
    normality_defect = float(np.linalg.norm(S @ S.T - S.T @ S, ord="fro"))

    controllability = np.column_stack(
        [np.linalg.matrix_power(S, k) @ B for k in range(N)]
    )
    observability = np.vstack(
        [C @ np.linalg.matrix_power(S, k) for k in range(N)]
    )

    controllability_rank = int(np.linalg.matrix_rank(controllability))
    observability_rank = int(np.linalg.matrix_rank(observability))
    max_recovery_error = float(np.max(np.abs(np.asarray(recovered) - INPUTS)))
    final_norm_after_silence = float(np.linalg.norm(x))

    result = {
        "gate": "travelling_trace",
        "cells": N,
        "inputs_oldest_to_newest": INPUTS.tolist(),
        "loaded_state_newest_to_oldest": loaded_state.tolist(),
        "single_terminal_sensor_recovered_oldest_to_newest": recovered,
        "eigenvalues": [float(v.real) for v in eig],
        "spectral_radius": spectral_radius,
        "nilpotent_power": N,
        "norm_S_power_N": nilpotent_residual,
        "normality_defect_frobenius": normality_defect,
        "controllability_rank": controllability_rank,
        "observability_rank_from_one_terminal_sensor_over_time": observability_rank,
        "max_recovery_error": max_recovery_error,
        "final_state_norm_after_N_silent_steps": final_norm_after_silence,
        "interpretation": (
            "The system stores and later emits eight past inputs exactly even though "
            "every eigenvalue of the autonomous update is zero. History is carried by "
            "directed transport through successive states; after eight silent steps the "
            "trace is exactly gone. Eigenvalue lifetime is therefore only one memory "
            "mechanism, not a complete theory of dynamical memory."
        ),
    }

    assert spectral_radius == 0.0
    assert nilpotent_residual == 0.0
    assert normality_defect > 0.0
    assert controllability_rank == N
    assert observability_rank == N
    assert max_recovery_error == 0.0
    assert final_norm_after_silence == 0.0

    out = Path("results") / "gate4_travelling_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
