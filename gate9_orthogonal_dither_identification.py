"""Gate 9: active orthogonal dither breaks one self/world confound.

Gate 8 showed an identifiability failure: if an external event is perfectly
correlated with a coarse self-command, simple command-response correlation
absorbs the external response into the learned negative image.

This gate adds a private, zero-mean micro-perturbation d to the coarse command a.
The material receives command amplitude

    c = a + eps * d

with a,d in {-1,+1}. Across the balanced four-point design

    (+,+), (+,-), (-,+), (-,-)

the coarse action and dither are exactly orthogonal. Suppose the external event
is correlated with the coarse action a but not with the private dither. In a
linear material,

    y = (a + eps d) S + a E

where S is the self echo and E is the external response. Then

    S_hat = (1 / (4 eps)) sum d y = S

exactly: the action-correlated external term cancels.

Attacker: if the external world also responds to the private dither,

    y = (a + eps d) S + (a + kappa d) E,

then S_hat = S + (kappa/eps)E. The ambiguity returns.

This is not a biological claim. It is an active-system-identification control
showing that carefully chosen interventions can create causal information that
passive correlation does not contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


NODES = 3
HORIZON = 8
EPSILON = 0.20
EXTERNAL_AMPLITUDE = 0.25
EXTERNAL_TIME = 2
EXTERNAL_PORT = 1
ATTACKER_KAPPA = 0.10
RECORD_ETA = 0.02
SELF_QUERY_EPISODES = 100
SEED = 97

MASK = np.ones((NODES, NODES), dtype=float) - np.eye(NODES, dtype=float)
PROBE = np.array([0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)], dtype=float)

A_AB = np.array(
    [
        [0.35, 0.25369472735764886, 0.28405603632996707],
        [0.23728892977981314, 0.4, 0.14919102987917615],
        [0.24879088308688233, 0.13296415272691187, 0.4],
    ],
    dtype=float,
)


def trajectory(command_amplitude: float, external_coefficient: float) -> np.ndarray:
    x = float(command_amplitude) * PROBE.copy()
    out = []

    for t in range(HORIZON):
        if t == EXTERNAL_TIME:
            event = np.zeros(NODES, dtype=float)
            event[EXTERNAL_PORT] = EXTERNAL_AMPLITUDE * float(external_coefficient)
            x += event

        x = A_AB @ x
        out.append(x.copy())

    return np.asarray(out, dtype=float).T


def exact_self_template() -> np.ndarray:
    return trajectory(1.0, 0.0)


def external_template() -> np.ndarray:
    return trajectory(0.0, 1.0)


def orthogonal_dither_estimate(kappa: float) -> np.ndarray:
    estimate = np.zeros((NODES, HORIZON), dtype=float)

    for a, d in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        command = a + EPSILON * d
        external = a + float(kappa) * d
        y = trajectory(command, external)
        estimate += d * y

    return estimate / (4.0 * EPSILON)


def self_only_material_drift(template: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    op = A_AB.copy()

    for _ in range(SELF_QUERY_EPISODES):
        a = 1.0 if rng.random() < 0.5 else -1.0
        actual = a * PROBE.copy()
        predicted_pre = a * PROBE.copy()

        for lag in range(HORIZON):
            pre = actual.copy()
            post = op @ pre
            predicted_post = a * template[:, lag]
            residual_pre = pre - predicted_pre
            residual_post = post - predicted_post

            op = np.clip(
                op
                + RECORD_ETA
                * np.outer(residual_post, residual_pre)
                * MASK,
                0.0,
                0.65,
            )

            actual = post
            predicted_pre = predicted_post

    return float(np.linalg.norm(op - A_AB))


def main() -> None:
    truth = exact_self_template()
    ext = external_template()

    clean = orthogonal_dither_estimate(kappa=0.0)
    attacked = orthogonal_dither_estimate(kappa=ATTACKER_KAPPA)

    clean_error = float(np.linalg.norm(clean - truth))
    attacked_error = float(np.linalg.norm(attacked - truth))
    predicted_attacked_error = float(
        np.linalg.norm((ATTACKER_KAPPA / EPSILON) * ext)
    )
    attacker_identity_error = float(
        np.linalg.norm(
            attacked - truth - (ATTACKER_KAPPA / EPSILON) * ext
        )
    )

    clean_drift = self_only_material_drift(clean, seed=SEED)
    attacked_drift = self_only_material_drift(attacked, seed=SEED)

    result = {
        "gate": "orthogonal_private_dither_identification",
        "seed": SEED,
        "nodes": NODES,
        "horizon": HORIZON,
        "epsilon": EPSILON,
        "external_amplitude": EXTERNAL_AMPLITUDE,
        "balanced_design": [
            {"coarse_action": 1, "private_dither": 1},
            {"coarse_action": 1, "private_dither": -1},
            {"coarse_action": -1, "private_dither": 1},
            {"coarse_action": -1, "private_dither": -1}
        ],
        "external_correlated_with_coarse_action_only": {
            "self_template_error_norm": clean_error,
            "self_only_100_query_material_drift": clean_drift
        },
        "attacker_external_also_correlated_with_private_dither": {
            "kappa": ATTACKER_KAPPA,
            "self_template_error_norm": attacked_error,
            "predicted_error_norm_kappa_over_epsilon_times_E": predicted_attacked_error,
            "identity_residual_norm": attacker_identity_error,
            "self_only_100_query_material_drift": attacked_drift
        },
        "interpretation": (
            "A balanced private dither creates causal leverage that coarse command "
            "correlation lacks. When the external disturbance tracks the coarse "
            "action but not the hidden dither, the self echo is recovered to machine "
            "precision despite perfect action/external correlation. If the external "
            "world also responds to the dither, the estimate is biased by exactly "
            "(kappa/epsilon) times the external response and false structural writing "
            "returns. Active intervention can therefore break some confounds, but no "
            "probe can separate causes that respond identically to every intervention "
            "available to the observer."
        )
    }

    assert clean_error < 1e-14
    assert clean_drift < 1e-14
    assert attacked_error > 0.05
    assert attacker_identity_error < 1e-14
    assert attacked_drift > 0.005

    out = Path("results") / "gate9_orthogonal_dither.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
