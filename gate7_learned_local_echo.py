"""Gate 7: learned local echo prediction, counted resources, and its failure mode.

Gate 6 used an exact negative image with privileged access to the current operator.
This gate removes that oracle.

Each material node owns only a short impulse-response template for ONE known
self-probe.  The template is learned from randomized +/- probe commands and the
node's own local response:

    h_j[t] <- h_j[t] + lr * a * (x_j[t] - a h_j[t])

where a in {-1,+1} is the outgoing-command copy.  The predictor never reads A.
For H lags and 3 nodes this costs exactly 3H slow predictor weights.  During an
episode it needs only the command sign and the current 3-value predicted state.

Stored-material plasticity uses local residual activity at an edge's endpoints.
Thus a bad self-prediction can itself write the material.  We test both sides:

1. A learned local echo model strongly suppresses self-query drift.
2. When the material legitimately changes, a fast echo model re-tracks it before
   much false writing occurs; a frozen/slow echo model corrupts more.
3. A persistent multiplicative prediction error creates systematic false writes
   even with no external event.  Exact cancellation is therefore a benchmark,
   not a free mechanism.

The AB/BA starting operators are the frozen Gate-5 learned materials.  Replacing
AB by BA in the change-of-echo attacker is an isolator for a valid material
change; this gate does not claim that the replacement itself was learned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 73
NODES = 3
HORIZON = 8
PREDICTOR_WEIGHTS = NODES * HORIZON
LOCAL_SENSOR_NOISE_STD = 0.001
CALIBRATION_EPISODES = 30
SELF_QUERY_EPISODES = 100
CHANGE_RECOVERY_EPISODES = 50
RECORD_ETA = 0.02
CALIBRATION_LR = 0.20

MASK = np.ones((NODES, NODES), dtype=float) - np.eye(NODES, dtype=float)
PROBE = np.array([0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)], dtype=float)

# Frozen Gate-5 materials.
A_AB = np.array(
    [
        [0.35, 0.25369472735764886, 0.28405603632996707],
        [0.23728892977981314, 0.4, 0.14919102987917615],
        [0.24879088308688233, 0.13296415272691187, 0.4],
    ],
    dtype=float,
)

A_BA = np.array(
    [
        [0.35, 0.28405603632996707, 0.25369472735764886],
        [0.24879088308688235, 0.4, 0.13296415272691192],
        [0.23728892977981317, 0.14919102987917615, 0.4],
    ],
    dtype=float,
)


def exact_template(operator: np.ndarray) -> np.ndarray:
    """Experimenter-only benchmark: unit-probe state at each lag."""
    x = PROBE.copy()
    trajectory = []
    for _ in range(HORIZON):
        x = operator @ x
        trajectory.append(x.copy())
    return np.asarray(trajectory, dtype=float).T


def material_update(
    operator: np.ndarray,
    residual_pre: np.ndarray,
    residual_post: np.ndarray,
) -> np.ndarray:
    delta = RECORD_ETA * np.outer(residual_post, residual_pre) * MASK
    return np.clip(operator + delta, 0.0, 0.65)


def probe_episode(
    operator: np.ndarray,
    template: np.ndarray,
    amplitude: float,
    predictor_lr: float,
    rng: np.random.Generator,
    record_plastic: bool,
    sensor_noise_std: float = LOCAL_SENSOR_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """One self-probe. Predictor sees command copy + local node response only."""
    op = operator.copy()
    h = template.copy()

    actual = float(amplitude) * PROBE.copy()
    predicted_pre = float(amplitude) * PROBE.copy()

    for lag in range(HORIZON):
        pre = actual.copy()
        post = op @ pre

        predicted_post = float(amplitude) * h[:, lag]
        residual_pre = pre - predicted_pre
        residual_post = post - predicted_post

        if record_plastic:
            op = material_update(op, residual_pre, residual_post)

        observed_local = post + rng.normal(
            0.0, sensor_noise_std, size=NODES
        )
        h[:, lag] += (
            predictor_lr
            * float(amplitude)
            * (observed_local - float(amplitude) * h[:, lag])
        )

        actual = post
        predicted_pre = predicted_post

    return op, h


def calibrate_predictor(
    episodes: int = CALIBRATION_EPISODES,
    seed: int = SEED,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    op = A_AB.copy()
    h = np.zeros((NODES, HORIZON), dtype=float)
    target = exact_template(A_AB)
    errors = []

    for _ in range(episodes):
        amplitude = 1.0 if rng.random() < 0.5 else -1.0
        op, h = probe_episode(
            op,
            h,
            amplitude,
            predictor_lr=CALIBRATION_LR,
            rng=rng,
            record_plastic=False,
        )
        errors.append(float(np.linalg.norm(h - target)))

    return h, errors


def self_query_drift(
    initial_template: np.ndarray,
    predictor_lr: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    op = A_AB.copy()
    h = initial_template.copy()

    for _ in range(SELF_QUERY_EPISODES):
        amplitude = 1.0 if rng.random() < 0.5 else -1.0
        op, h = probe_episode(
            op,
            h,
            amplitude,
            predictor_lr=predictor_lr,
            rng=rng,
            record_plastic=True,
        )

    drift = float(np.linalg.norm(op - A_AB))
    predictor_error = float(np.linalg.norm(h - exact_template(op)))
    return drift, predictor_error


def recover_after_valid_material_change(
    old_template: np.ndarray,
    predictor_lr: float,
    seed: int,
) -> dict:
    """AB echo model is stale because the material was validly changed to BA."""
    rng = np.random.default_rng(seed)
    op = A_BA.copy()
    h = old_template.copy()
    samples = []

    checkpoints = {1, 5, 20, CHANGE_RECOVERY_EPISODES}
    for episode in range(1, CHANGE_RECOVERY_EPISODES + 1):
        amplitude = 1.0 if rng.random() < 0.5 else -1.0
        op, h = probe_episode(
            op,
            h,
            amplitude,
            predictor_lr=predictor_lr,
            rng=rng,
            record_plastic=True,
        )

        if episode in checkpoints:
            samples.append(
                {
                    "episode": episode,
                    "material_drift_from_valid_BA": float(
                        np.linalg.norm(op - A_BA)
                    ),
                    "local_predictor_error": float(
                        np.linalg.norm(h - exact_template(op))
                    ),
                }
            )

    return {
        "predictor_lr": predictor_lr,
        "checkpoints": samples,
        "final_material_drift_from_valid_BA": float(np.linalg.norm(op - A_BA)),
        "final_local_predictor_error": float(
            np.linalg.norm(h - exact_template(op))
        ),
    }


def scale_error_attack(scale: float, episodes: int, seed: int) -> float:
    """Persistent correlated prediction error with no external event."""
    rng = np.random.default_rng(seed)
    op = A_BA.copy()
    perfect = exact_template(A_BA)

    for _ in range(episodes):
        amplitude = 1.0 if rng.random() < 0.5 else -1.0
        actual = float(amplitude) * PROBE.copy()
        predicted_pre = float(amplitude) * PROBE.copy()

        for lag in range(HORIZON):
            pre = actual.copy()
            post = op @ pre
            predicted_post = float(amplitude) * scale * perfect[:, lag]
            op = material_update(
                op,
                pre - predicted_pre,
                post - predicted_post,
            )
            actual = post
            predicted_pre = predicted_post

    return float(np.linalg.norm(op - A_BA))


def main() -> None:
    learned_h, calibration_errors = calibrate_predictor()

    no_predictor_drift, _ = self_query_drift(
        np.zeros((NODES, HORIZON), dtype=float),
        predictor_lr=0.0,
        seed=SEED + 1,
    )
    learned_drift, learned_final_error = self_query_drift(
        learned_h,
        predictor_lr=CALIBRATION_LR,
        seed=SEED + 1,
    )

    change_recovery = {
        "frozen": recover_after_valid_material_change(
            learned_h, predictor_lr=0.0, seed=SEED + 2
        ),
        "slow": recover_after_valid_material_change(
            learned_h, predictor_lr=0.02, seed=SEED + 2
        ),
        "fast": recover_after_valid_material_change(
            learned_h, predictor_lr=0.60, seed=SEED + 2
        ),
    }

    exact_error_drift = scale_error_attack(
        scale=1.0, episodes=SELF_QUERY_EPISODES, seed=SEED + 3
    )
    ten_percent_error_drift = scale_error_attack(
        scale=0.90, episodes=SELF_QUERY_EPISODES, seed=SEED + 3
    )

    result = {
        "gate": "learned_local_echo_predictor",
        "seed": SEED,
        "nodes": NODES,
        "horizon": HORIZON,
        "predictor_weights": PREDICTOR_WEIGHTS,
        "temporary_predicted_state_values": NODES,
        "outgoing_command_copy_values": 1,
        "local_sensor_noise_std": LOCAL_SENSOR_NOISE_STD,
        "calibration_episodes": CALIBRATION_EPISODES,
        "calibration_lr": CALIBRATION_LR,
        "calibration_error_norm": {
            "after_1": calibration_errors[0],
            "after_5": calibration_errors[4],
            "after_10": calibration_errors[9],
            "after_20": calibration_errors[19],
            "after_30": calibration_errors[29],
        },
        "self_only_100_queries": {
            "no_predictor_material_drift": no_predictor_drift,
            "learned_predictor_material_drift": learned_drift,
            "learned_predictor_final_error": learned_final_error,
            "drift_reduction_factor": no_predictor_drift / max(learned_drift, 1e-30),
        },
        "valid_AB_to_BA_material_change_with_stale_echo_model": change_recovery,
        "correlated_prediction_error_attack_100_queries": {
            "perfect_scale_1_material_drift": exact_error_drift,
            "scale_0.90_material_drift": ten_percent_error_drift,
        },
        "interpretation": (
            "A counted 24-weight local impulse-response predictor can learn the "
            "self-generated echo from randomized outgoing-command copies without "
            "reading the material matrix. Once learned, it suppresses self-query "
            "writing by orders of magnitude. After a valid material change, a fast "
            "echo predictor re-tracks the new response before much false structural "
            "writing accumulates, whereas a frozen/slow predictor corrupts more. "
            "However, persistent temporally correlated prediction error still writes "
            "the material even when no external event occurred. Exact Gate-6 "
            "cancellation was therefore an ideal benchmark, not a solved mechanism."
        ),
    }

    assert PREDICTOR_WEIGHTS == 24
    assert calibration_errors[-1] < calibration_errors[0] * 0.02
    assert no_predictor_drift > learned_drift * 1000.0

    frozen_drift = change_recovery["frozen"]["final_material_drift_from_valid_BA"]
    fast_drift = change_recovery["fast"]["final_material_drift_from_valid_BA"]
    assert frozen_drift > fast_drift * 20.0
    assert change_recovery["fast"]["final_local_predictor_error"] < 0.01

    assert exact_error_drift < 1e-14
    assert ten_percent_error_drift > 1e-4

    out = Path("results") / "gate7_learned_local_echo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
