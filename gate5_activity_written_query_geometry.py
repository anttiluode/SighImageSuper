"""Gate 5: activity-written material + query geometry.

This gate fixes a limitation of Gate 3. Gate 3 wrote order directly from the
supplied cue and a cue eligibility trace. Here the material only changes from
the activity that actually propagates through the material.

Protocol
--------
1. TEACH AB or BA in an initially mirror-symmetric three-port material.
2. Local plasticity on edge i -> j uses pre_i(t) * post_j(t+1), where post is
   produced by the current propagation operator.
3. SILENCE every fast state exactly to zero.
4. Freeze the learned material.
5. QUESTION it through a bounded observer that can listen only at neutral port P.

The key test is geometric. A probe launched at P is exactly blind to AB vs BA
at P because the learned materials are mirror images that leave P fixed.
A differential unit-energy branch probe is informative at that same sensor.

A propagation-off teaching control must produce no material update at all. This
shows that the material's own evolving activity participates in writing memory.

The optimal probe below is an analytical benchmark: it is computed from the two
candidate frozen operators. It is NOT yet an autonomous probe-selection policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 47
TRIALS = 10_000

P, A_PORT, B_PORT = 0, 1, 2

ETA = 0.04
TEACH_EPOCHS = 10
RELAX_STEPS = 3
GAP_STEPS = 1
PROBE_HORIZON = 4
READOUT_NOISE_STD = 0.01

PLASTIC_MASK = np.ones((3, 3), dtype=float) - np.eye(3, dtype=float)


def baseline_operator() -> np.ndarray:
    """Mirror-symmetric stable virgin material. A[j,i] means i -> j."""
    return np.array(
        [
            [0.35, 0.12, 0.12],
            [0.18, 0.40, 0.04],
            [0.18, 0.04, 0.40],
        ],
        dtype=float,
    )


def cue_vector(cue: str, amplitude: float) -> np.ndarray:
    u = np.zeros(3, dtype=float)
    u[A_PORT if cue == "A" else B_PORT] = float(amplitude)
    return u


def local_activity_update(
    operator: np.ndarray,
    pre_state: np.ndarray,
    post_state: np.ndarray,
) -> np.ndarray:
    """Local pre/post rule: edge i->j sees only pre_i and propagated post_j."""
    delta = ETA * np.outer(post_state, pre_state) * PLASTIC_MASK
    return np.clip(operator + delta, 0.0, 0.65)


def teach(
    order: str,
    amplitude: float = 1.0,
    plastic: bool = True,
    propagation_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    operator = baseline_operator().copy()

    for _ in range(TEACH_EPOCHS):
        state = np.zeros(3, dtype=float)

        for cue in order:
            state = state + cue_vector(cue, amplitude)

            for _ in range(RELAX_STEPS):
                pre = state.copy()
                post = (
                    operator @ pre
                    if propagation_enabled
                    else np.zeros_like(pre)
                )

                if plastic:
                    operator = local_activity_update(operator, pre, post)

                state = post

            for _ in range(GAP_STEPS):
                pre = state.copy()
                post = (
                    operator @ pre
                    if propagation_enabled
                    else np.zeros_like(pre)
                )

                if plastic:
                    operator = local_activity_update(operator, pre, post)

                state = post

    # SILENCE. No fast activity survives into the question phase.
    state = np.zeros(3, dtype=float)
    return operator, state


def response_operator(operator: np.ndarray, horizon: int = PROBE_HORIZON) -> np.ndarray:
    """H maps an initial probe p to P-sensor samples over the next horizon."""
    sensor = np.array([[1.0, 0.0, 0.0]], dtype=float)
    rows = []
    power = np.eye(3, dtype=float)

    for _ in range(horizon):
        power = power @ operator
        rows.append((sensor @ power).ravel())

    return np.stack(rows, axis=0)


def gaussian_discriminant(
    y: np.ndarray,
    mean_ab: np.ndarray,
    mean_ba: np.ndarray,
) -> str:
    """Equal-covariance nearest-mean linear discriminant."""
    direction = mean_ab - mean_ba
    threshold = 0.5 * (
        float(mean_ab @ mean_ab) - float(mean_ba @ mean_ba)
    )
    score = float(direction @ y) - threshold
    return "AB" if score > 0.0 else "BA"


def eigen_pairs(operator: np.ndarray) -> list[list[float]]:
    vals = np.sort_complex(np.linalg.eigvals(operator))
    return [[float(v.real), float(v.imag)] for v in vals]


def main() -> None:
    operator_ab, silent_ab = teach("AB")
    operator_ba, silent_ba = teach("BA")

    H_ab = response_operator(operator_ab)
    H_ba = response_operator(operator_ba)
    delta_H = H_ab - H_ba

    center_probe = np.array([1.0, 0.0, 0.0], dtype=float)
    branch_a_probe = np.array([0.0, 1.0, 0.0], dtype=float)

    # Best unit-energy probe for common isotropic sensor noise.
    _, singular_values, vh = np.linalg.svd(delta_H, full_matrices=False)
    optimal_probe = vh[0].copy()

    # Fix only an arbitrary sign convention, not the information content.
    if optimal_probe[A_PORT] < 0.0:
        optimal_probe *= -1.0

    center_signal = delta_H @ center_probe
    branch_a_signal = delta_H @ branch_a_probe
    optimal_signal = delta_H @ optimal_probe

    # The write must disappear when the material cannot propagate activity.
    operator_no_prop, silent_no_prop = teach(
        "AB",
        plastic=True,
        propagation_enabled=False,
    )
    no_prop_material_change = float(
        np.linalg.norm(operator_no_prop - baseline_operator())
    )

    eig_ab = np.sort_complex(np.linalg.eigvals(operator_ab))
    eig_ba = np.sort_complex(np.linalg.eigvals(operator_ba))
    eig_difference = float(np.max(np.abs(eig_ab - eig_ba)))

    rng = np.random.default_rng(SEED)
    center_correct = 0
    optimal_correct = 0

    nominal_center_ab = H_ab @ center_probe
    nominal_center_ba = H_ba @ center_probe
    nominal_opt_ab = H_ab @ optimal_probe
    nominal_opt_ba = H_ba @ optimal_probe

    # Vary overall teaching strength without changing the identity/count of cues.
    # The fixed analytical probe/readout is not re-fit on each trial.
    for _ in range(TRIALS):
        order = "AB" if int(rng.integers(0, 2)) == 0 else "BA"
        amplitude = float(rng.uniform(0.8, 1.2))
        operator, silent = teach(order, amplitude=amplitude)
        assert np.all(silent == 0.0)

        H_trial = response_operator(operator)

        y_center = (
            H_trial @ center_probe
            + rng.normal(0.0, READOUT_NOISE_STD, size=PROBE_HORIZON)
        )
        y_opt = (
            H_trial @ optimal_probe
            + rng.normal(0.0, READOUT_NOISE_STD, size=PROBE_HORIZON)
        )

        center_correct += int(
            gaussian_discriminant(
                y_center,
                nominal_center_ab,
                nominal_center_ba,
            )
            == order
        )
        optimal_correct += int(
            gaussian_discriminant(
                y_opt,
                nominal_opt_ab,
                nominal_opt_ba,
            )
            == order
        )

    result = {
        "gate": "activity_written_query_geometry",
        "seed": SEED,
        "trials": TRIALS,
        "eta": ETA,
        "teach_epochs": TEACH_EPOCHS,
        "relax_steps": RELAX_STEPS,
        "gap_steps": GAP_STEPS,
        "probe_horizon": PROBE_HORIZON,
        "readout_noise_std": READOUT_NOISE_STD,
        "silent_fast_state_AB": silent_ab.tolist(),
        "silent_fast_state_BA": silent_ba.tolist(),
        "operator_AB": operator_ab.tolist(),
        "operator_BA": operator_ba.tolist(),
        "eigenvalues_AB_real_imag": eigen_pairs(operator_ab),
        "eigenvalues_BA_real_imag": eigen_pairs(operator_ba),
        "eigenvalue_set_max_difference": eig_difference,
        "spectral_radius": float(np.max(np.abs(eig_ab))),
        "propagation_disabled_material_change_norm": no_prop_material_change,
        "propagation_disabled_silent_state": silent_no_prop.tolist(),
        "delta_H": delta_H.tolist(),
        "center_probe": center_probe.tolist(),
        "center_probe_signal_norm_at_P_sensor": float(
            np.linalg.norm(center_signal)
        ),
        "branch_A_probe_signal_norm_at_P_sensor": float(
            np.linalg.norm(branch_a_signal)
        ),
        "optimal_unit_energy_probe": optimal_probe.tolist(),
        "optimal_probe_signal_norm_at_P_sensor": float(
            np.linalg.norm(optimal_signal)
        ),
        "optimal_probe_top_singular_value": float(singular_values[0]),
        "center_probe_noisy_accuracy": center_correct / TRIALS,
        "optimal_probe_noisy_accuracy": optimal_correct / TRIALS,
        "interpretation": (
            "Order is now written by local pre/post activity generated by the "
            "material's own propagation. Disabling propagation eliminates the "
            "write. After complete fast-state silence, a P->P question is exactly "
            "blind by mirror symmetry, while a differential branch probe makes the "
            "same P sensor strongly informative. Memory therefore depends jointly "
            "on learned material and query/readout geometry; probe choice can reveal "
            "a distinction that passive observation cannot see."
        ),
    }

    # Mechanism receipts.
    assert np.all(silent_ab == 0.0)
    assert np.all(silent_ba == 0.0)
    assert no_prop_material_change == 0.0
    assert eig_difference < 1e-12
    assert float(np.max(np.abs(eig_ab))) < 1.0

    # Exact mirror ambiguity at the neutral port.
    assert np.linalg.norm(center_signal) < 1e-14

    # Probe geometry opens an otherwise hidden direction.
    assert np.linalg.norm(branch_a_signal) > 0.03
    assert np.linalg.norm(optimal_signal) > 0.05

    # Same noise/budget: center is chance, differential query is robust.
    assert 0.47 < result["center_probe_noisy_accuracy"] < 0.53
    assert result["optimal_probe_noisy_accuracy"] > 0.95

    out = Path("results") / "gate5_activity_query.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
