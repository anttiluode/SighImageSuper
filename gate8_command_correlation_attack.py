"""Gate 8: randomized command-correlation learning and its ambiguity attack.

Gate 7 replaced the exact self-model with a counted local impulse-response
predictor. A natural way to learn that predictor without reading the material
matrix is to randomize the sign of a known self-probe and correlate each local
response with the outgoing command copy.

For a probe sign a in {-1,+1}, each node/lag learns the running average of

    a * x

If unrelated external activity is statistically independent of a, its
contribution averages toward zero and the estimate approaches the self-generated
echo.

But if an external event is itself correlated with the command, the same
estimator cannot know whether that command-correlated component was caused by
self or world. It absorbs the external response into the negative image.

This gate demonstrates both facts with the frozen Gate-5 AB material:

  * independent external disturbances -> learned echo converges to true self;
  * command-correlated disturbances -> learned echo incorporates the world;
  * the corrupted predictor then creates false residuals on pure self-probes and
    can cancel a genuinely external event that has the learned correlation.

That is an identifiability limit, not merely a tuning failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 83
NODES = 3
HORIZON = 8
TRAIN_EPISODES = 5000
PREDICTOR_LR = "running_mean_1_over_n"
EXTERNAL_AMPLITUDE = 0.25
EXTERNAL_TIME = 2
EXTERNAL_PORT = 1
SELF_DRIFT_EPISODES = 100
RECORD_ETA = 0.02

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


def exact_self_template() -> np.ndarray:
    x = PROBE.copy()
    out = []
    for _ in range(HORIZON):
        x = A_AB @ x
        out.append(x.copy())
    return np.asarray(out, dtype=float).T


def observed_trajectory(
    probe_sign: float,
    external_sign: float | None,
) -> np.ndarray:
    x = float(probe_sign) * PROBE.copy()
    out = []

    for t in range(HORIZON):
        if external_sign is not None and t == EXTERNAL_TIME:
            event = np.zeros(NODES, dtype=float)
            event[EXTERNAL_PORT] = EXTERNAL_AMPLITUDE * float(external_sign)
            x += event

        x = A_AB @ x
        out.append(x.copy())

    return np.asarray(out, dtype=float).T


def train_predictor(
    correlated_external: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.zeros((NODES, HORIZON), dtype=float)

    for index in range(TRAIN_EPISODES):
        a = 1.0 if rng.random() < 0.5 else -1.0
        if correlated_external:
            z = a
        else:
            z = 1.0 if rng.random() < 0.5 else -1.0

        observed = observed_trajectory(a, z)
        target = a * observed
        n = index + 1
        h += (target - h) / n

    return h


def self_only_material_drift(template: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    op = A_AB.copy()

    for _ in range(SELF_DRIFT_EPISODES):
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


def residual_norm(
    template: np.ndarray,
    probe_sign: float,
    external_sign: float | None,
) -> float:
    observed = observed_trajectory(probe_sign, external_sign)
    predicted = float(probe_sign) * template
    return float(np.linalg.norm(observed - predicted))


def main() -> None:
    truth = exact_self_template()
    independent = train_predictor(False, seed=SEED)
    correlated = train_predictor(True, seed=SEED)

    independent_error = float(np.linalg.norm(independent - truth))
    correlated_error = float(np.linalg.norm(correlated - truth))

    independent_self_drift = self_only_material_drift(
        independent, seed=SEED + 1
    )
    correlated_self_drift = self_only_material_drift(
        correlated, seed=SEED + 1
    )

    external_residual_independent_model = residual_norm(
        independent, probe_sign=1.0, external_sign=1.0
    )
    external_residual_correlated_model = residual_norm(
        correlated, probe_sign=1.0, external_sign=1.0
    )

    pure_self_residual_independent_model = residual_norm(
        independent, probe_sign=1.0, external_sign=None
    )
    pure_self_residual_correlated_model = residual_norm(
        correlated, probe_sign=1.0, external_sign=None
    )

    result = {
        "gate": "command_correlation_attack",
        "seed": SEED,
        "nodes": NODES,
        "horizon": HORIZON,
        "predictor_weights": NODES * HORIZON,
        "training_episodes": TRAIN_EPISODES,
        "predictor_lr": PREDICTOR_LR,
        "external_amplitude": EXTERNAL_AMPLITUDE,
        "external_time": EXTERNAL_TIME,
        "external_port": EXTERNAL_PORT,
        "learned_template_error_norm": {
            "external_independent_of_command": independent_error,
            "external_perfectly_correlated_with_command": correlated_error,
        },
        "pure_self_100_query_material_drift": {
            "predictor_trained_with_independent_external": independent_self_drift,
            "predictor_trained_with_correlated_external": correlated_self_drift,
        },
        "residual_norm_on_command_correlated_external_event": {
            "predictor_trained_with_independent_external": (
                external_residual_independent_model
            ),
            "predictor_trained_with_correlated_external": (
                external_residual_correlated_model
            ),
        },
        "residual_norm_on_pure_self_probe": {
            "predictor_trained_with_independent_external": (
                pure_self_residual_independent_model
            ),
            "predictor_trained_with_correlated_external": (
                pure_self_residual_correlated_model
            ),
        },
        "interpretation": (
            "Randomized +/- self-probes let a local correlation learner recover the "
            "self echo when external disturbances are independent of the command. "
            "When the external event is perfectly command-correlated, the same "
            "estimator absorbs that external response into its negative image. The "
            "corrupted predictor then produces large false residuals on pure self "
            "queries and nearly cancels the genuinely external correlated event. "
            "Self/world separation therefore needs assumptions or extra causal "
            "interventions beyond command correlation alone."
        ),
    }

    assert independent_error < 1e-3
    assert correlated_error > 0.1
    assert correlated_self_drift > independent_self_drift * 1000.0
    assert external_residual_independent_model > 0.1
    assert external_residual_correlated_model < 1e-10
    assert pure_self_residual_correlated_model > 0.1

    out = Path("results") / "gate8_command_correlation_attack.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
