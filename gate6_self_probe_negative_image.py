"""Gate 6: protect material from self-interrogation without freezing the world.

Gate 5 established that activity-written material can remember AB versus BA after
all fast state is erased, and that probe geometry determines whether the memory
is visible. This gate asks the next Jello/ThinkingJello question:

    Can a material ignore plasticity caused by its own question while remaining
    plastic to new external evidence that arrives during that question?

The toy is intentionally linear so the mechanism can be audited exactly.

Three policies are compared during a query episode:
  * always_on: every actual pre/post state updates the material;
  * global_freeze: no update is allowed while querying;
  * negative_image: keep a temporary prediction of the self-generated probe
    response from the outgoing-command copy and update only from the residual.

The predictor carries no long-term history. It is reset every episode and uses
only the current operator plus the known self-probe command.

In this linear isolator, if an external BA event arrives during the query, the
negative-image residual follows exactly the state that the external event would
have produced by itself. Therefore its material update should match an
external-only reference to numerical precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate5_activity_written_query_geometry import (
    A_PORT,
    ETA,
    PLASTIC_MASK,
    cue_vector,
    response_operator,
    teach,
)


SEED = 61
SELF_ONLY_QUERIES = 20
MAX_REVERSAL_EPISODES = 10
EXTERNAL_GAP = 4


def nominal_probe_and_score():
    """Return the Gate-5 optimal probe and a fixed AB-vs-BA score function."""
    op_ab, _ = teach("AB")
    op_ba, _ = teach("BA")

    H_ab = response_operator(op_ab)
    H_ba = response_operator(op_ba)
    delta = H_ab - H_ba

    _, _, vh = np.linalg.svd(delta, full_matrices=False)
    probe = vh[0].copy()
    if probe[A_PORT] < 0.0:
        probe *= -1.0

    mu_ab = H_ab @ probe
    mu_ba = H_ba @ probe
    direction = mu_ab - mu_ba
    threshold = 0.5 * (
        float(mu_ab @ mu_ab) - float(mu_ba @ mu_ba)
    )

    def score(operator: np.ndarray) -> float:
        y = response_operator(operator) @ probe
        return float(direction @ y) - threshold

    return probe, score, op_ab, op_ba


def update_from(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    return ETA * np.outer(post, pre) * PLASTIC_MASK


def query_episode(
    operator: np.ndarray,
    probe: np.ndarray,
    policy: str,
    external_order: str | None = None,
) -> np.ndarray:
    """Run one self-query, optionally with a two-cue external event overlapping."""
    op = operator.copy()
    actual = np.zeros(3, dtype=float)
    predicted_self = np.zeros(3, dtype=float)

    external = {}
    if external_order is not None:
        external[1] = cue_vector(external_order[0], 1.0)
        external[1 + EXTERNAL_GAP] = cue_vector(external_order[1], 1.0)

    total_steps = 1 + EXTERNAL_GAP + 3 + 1

    for t in range(total_steps):
        if t == 0:
            actual += probe
            predicted_self += probe

        if t in external:
            actual += external[t]

        pre = actual.copy()
        self_pre = predicted_self.copy()

        post = op @ pre
        self_post = op @ self_pre

        if policy == "always_on":
            op = np.clip(op + update_from(pre, post), 0.0, 0.65)

        elif policy == "global_freeze":
            pass

        elif policy == "negative_image":
            residual_pre = pre - self_pre
            residual_post = post - self_post
            op = np.clip(
                op + update_from(residual_pre, residual_post),
                0.0,
                0.65,
            )
        else:
            raise ValueError(policy)

        actual = post
        predicted_self = self_post

    return op


def external_only_episode(
    operator: np.ndarray,
    external_order: str,
) -> np.ndarray:
    """Reference: same external event/timing, but no self-generated probe."""
    op = operator.copy()
    state = np.zeros(3, dtype=float)

    external = {
        1: cue_vector(external_order[0], 1.0),
        1 + EXTERNAL_GAP: cue_vector(external_order[1], 1.0),
    }
    total_steps = 1 + EXTERNAL_GAP + 3 + 1

    for t in range(total_steps):
        if t in external:
            state += external[t]

        pre = state.copy()
        post = op @ pre
        op = np.clip(op + update_from(pre, post), 0.0, 0.65)
        state = post

    return op


def first_reversal_episode(
    start: np.ndarray,
    probe: np.ndarray,
    score,
    policy: str,
) -> tuple[int | None, list[float], np.ndarray]:
    op = start.copy()
    scores = []

    for episode in range(1, MAX_REVERSAL_EPISODES + 1):
        op = query_episode(
            op,
            probe,
            policy=policy,
            external_order="BA",
        )
        s = score(op)
        scores.append(float(s))
        if s < 0.0:
            return episode, scores, op

    return None, scores, op


def main() -> None:
    probe, score, op_ab, _ = nominal_probe_and_score()
    initial_score = score(op_ab)

    self_only = {}
    for policy in ("always_on", "global_freeze", "negative_image"):
        op = op_ab.copy()
        for _ in range(SELF_ONLY_QUERIES):
            op = query_episode(op, probe, policy=policy)

        self_only[policy] = {
            "material_drift_norm": float(np.linalg.norm(op - op_ab)),
            "memory_score_after_queries": float(score(op)),
        }

    # One overlapping external episode gives the cleanest exact receipt.
    ext_reference = external_only_episode(op_ab, "BA")
    overlap = {}
    for policy in ("always_on", "global_freeze", "negative_image"):
        op = query_episode(
            op_ab,
            probe,
            policy=policy,
            external_order="BA",
        )
        overlap[policy] = {
            "distance_from_external_only_reference": float(
                np.linalg.norm(op - ext_reference)
            ),
            "memory_score_after_one_overlap": float(score(op)),
        }

    reversal = {}
    for policy in ("always_on", "global_freeze", "negative_image"):
        flip, scores, final_op = first_reversal_episode(
            op_ab,
            probe,
            score,
            policy,
        )
        reversal[policy] = {
            "first_episode_scored_as_BA": flip,
            "scores": scores,
            "final_material_drift_norm": float(
                np.linalg.norm(final_op - op_ab)
            ),
        }

    # Verify negative-image overlap remains identical to external-only for
    # multiple sequential reversal episodes, not just one.
    op_residual = op_ab.copy()
    op_external = op_ab.copy()
    for _ in range(3):
        op_residual = query_episode(
            op_residual,
            probe,
            policy="negative_image",
            external_order="BA",
        )
        op_external = external_only_episode(op_external, "BA")

    three_episode_equivalence = float(
        np.linalg.norm(op_residual - op_external)
    )

    result = {
        "gate": "self_probe_negative_image",
        "seed": SEED,
        "self_only_queries": SELF_ONLY_QUERIES,
        "max_reversal_episodes": MAX_REVERSAL_EPISODES,
        "probe": probe.tolist(),
        "initial_AB_memory_score": float(initial_score),
        "self_only": self_only,
        "overlapping_external_BA": overlap,
        "reversal_during_queries": reversal,
        "negative_image_vs_external_only_after_3_episodes_norm": (
            three_episode_equivalence
        ),
        "temporary_predictor_state_is_reset_each_episode": True,
        "interpretation": (
            "Always-on plasticity lets the act of questioning reshape the stored "
            "material. Global freezing protects the memory but also blocks new "
            "external evidence that arrives during the question. In this linear "
            "isolator, subtracting the predicted self-generated probe response "
            "makes self-only interrogation produce zero material drift while an "
            "overlapping external BA event updates the material exactly as it would "
            "without the probe. This is an exact negative-image/corollary-discharge "
            "control, not yet a learned biological mechanism."
        ),
    }

    # Self-only interrogation.
    assert self_only["always_on"]["material_drift_norm"] > 0.1
    assert self_only["global_freeze"]["material_drift_norm"] == 0.0
    assert self_only["negative_image"]["material_drift_norm"] < 1e-14

    # Hard freeze misses external evidence; residual isolates it.
    assert overlap["global_freeze"][
        "distance_from_external_only_reference"
    ] > 0.05
    assert overlap["always_on"][
        "distance_from_external_only_reference"
    ] > 1e-3
    assert overlap["negative_image"][
        "distance_from_external_only_reference"
    ] < 1e-14

    assert three_episode_equivalence < 1e-14

    # The protected-but-plastic policy can actually reverse the answer.
    assert reversal["global_freeze"]["first_episode_scored_as_BA"] is None
    assert reversal["negative_image"]["first_episode_scored_as_BA"] is not None
    assert reversal["negative_image"]["first_episode_scored_as_BA"] <= 4

    out = Path("results") / "gate6_self_probe_negative_image.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
