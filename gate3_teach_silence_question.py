"""Gate 3: teach -> silence -> question.

A tiny three-node material isolates a mechanism that the image loop itself does
not contain yet: history can survive in the operator even after every fast state
variable has been erased.

The teaching rule is intentionally transparent and local in time.  If cue i was
active on the previous teaching step and cue j is active now, strengthen the
directed material connection i -> j.  Thus AB and BA write mirror-image
operators.

After teaching we erase the fast activity and the eligibility trace, freeze the
material, and inject exactly the same neutral probe.  A two-sensor readout must
recover the earlier order from the probe response.

Two controls matter:
  * freeze plasticity during teaching;
  * restore the virgin material before probing.

In both controls the order information must disappear.

The AB and BA learned operators are related by swapping the A/B coordinates, so
they have the same eigenvalues.  This is deliberate: remembered structure need
not be represented by a change in decay times/eigenvalues alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 23
TRIALS = 10_000
ETA = 0.18
PROBE_STEPS = 3
READOUT_NOISE_STD = 0.02

# State coordinates: [neutral probe port P, cue/sensor A, cue/sensor B].
CUES = {
    "A": np.array([0.0, 1.0, 0.0], dtype=float),
    "B": np.array([0.0, 0.0, 1.0], dtype=float),
}


def baseline_operator() -> np.ndarray:
    """Stable virgin material.  A[j, i] is transport i -> j."""
    return np.array(
        [
            [0.45, 0.08, 0.08],
            [0.25, 0.45, 0.03],
            [0.25, 0.03, 0.45],
        ],
        dtype=float,
    )


def teach(
    order: str,
    amplitudes=(1.0, 1.0),
    plastic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teach AB or BA, then erase every fast/temporary teaching variable."""
    A = baseline_operator().copy()
    fast_state = np.zeros(3, dtype=float)
    eligibility = np.zeros(3, dtype=float)

    for cue, amp in zip(order, amplitudes):
        u = float(amp) * CUES[cue]

        # Fast activity is allowed during teaching, but it will be erased before
        # the question is asked.
        fast_state = A @ fast_state + u

        # Minimal temporal-order plasticity: previous cue -> current cue.
        if plastic:
            A += ETA * np.outer(u, eligibility)

        eligibility = u.copy()

    # SILENCE: no activity or temporary trace may carry the answer.
    fast_state[:] = 0.0
    eligibility[:] = 0.0
    return A, fast_state, eligibility


def probe(
    A: np.ndarray,
    steps: int = PROBE_STEPS,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze A and ask both materials exactly the same neutral question."""
    x = np.array([1.0, 0.0, 0.0], dtype=float)
    for _ in range(steps):
        x = A @ x

    # Bounded observer: only the two terminal sensors are visible.
    y = x[[1, 2]].copy()
    if rng is not None and noise_std > 0.0:
        y += rng.normal(0.0, noise_std, size=2)
    return y, x


def sorted_eigen_pairs(A: np.ndarray) -> list[list[float]]:
    vals = np.sort_complex(np.linalg.eigvals(A))
    return [[float(z.real), float(z.imag)] for z in vals]


def classify(y: np.ndarray) -> str:
    # AB strengthens A -> B, so the neutral probe eventually reaches B more.
    return "AB" if y[1] > y[0] else "BA"


def main() -> None:
    A_ab, fast_ab, elig_ab = teach("AB")
    A_ba, fast_ba, elig_ba = teach("BA")

    y_ab, x_ab = probe(A_ab)
    y_ba, x_ba = probe(A_ba)

    eig_ab = np.sort_complex(np.linalg.eigvals(A_ab))
    eig_ba = np.sort_complex(np.linalg.eigvals(A_ba))
    eig_max_difference = float(np.max(np.abs(eig_ab - eig_ba)))

    rng = np.random.default_rng(SEED)
    trained_correct = 0
    frozen_correct = 0
    restored_correct = 0
    signed_margins = []

    for _ in range(TRIALS):
        order = "AB" if int(rng.integers(0, 2)) == 0 else "BA"
        amplitudes = tuple(rng.uniform(0.7, 1.3, size=2))
        target_sign = 1.0 if order == "AB" else -1.0

        learned_A, fast_state, eligibility = teach(
            order, amplitudes=amplitudes, plastic=True
        )
        assert np.all(fast_state == 0.0)
        assert np.all(eligibility == 0.0)

        y, _ = probe(
            learned_A,
            noise_std=READOUT_NOISE_STD,
            rng=rng,
        )
        trained_correct += int(classify(y) == order)
        signed_margins.append(target_sign * float(y[1] - y[0]))

        # Control 1: teaching activity occurs, but the material cannot change.
        frozen_A, _, _ = teach(order, amplitudes=amplitudes, plastic=False)
        y_frozen, _ = probe(
            frozen_A,
            noise_std=READOUT_NOISE_STD,
            rng=rng,
        )
        frozen_correct += int(classify(y_frozen) == order)

        # Control 2: teach normally, then restore virgin conductances.
        y_restored, _ = probe(
            baseline_operator(),
            noise_std=READOUT_NOISE_STD,
            rng=rng,
        )
        restored_correct += int(classify(y_restored) == order)

    trained_accuracy = trained_correct / TRIALS
    frozen_accuracy = frozen_correct / TRIALS
    restored_accuracy = restored_correct / TRIALS

    result = {
        "gate": "teach_silence_question",
        "seed": SEED,
        "trials": TRIALS,
        "eta": ETA,
        "probe_steps": PROBE_STEPS,
        "readout_noise_std": READOUT_NOISE_STD,
        "fast_state_after_silence_AB": fast_ab.tolist(),
        "fast_state_after_silence_BA": fast_ba.tolist(),
        "eligibility_after_silence_AB": elig_ab.tolist(),
        "eligibility_after_silence_BA": elig_ba.tolist(),
        "operator_AB": A_ab.tolist(),
        "operator_BA": A_ba.tolist(),
        "eigenvalues_AB_real_imag": sorted_eigen_pairs(A_ab),
        "eigenvalues_BA_real_imag": sorted_eigen_pairs(A_ba),
        "eigenvalue_set_max_difference": eig_max_difference,
        "spectral_radius": float(np.max(np.abs(eig_ab))),
        "neutral_probe_terminal_response_AB": y_ab.tolist(),
        "neutral_probe_terminal_response_BA": y_ba.tolist(),
        "neutral_probe_full_state_AB": x_ab.tolist(),
        "neutral_probe_full_state_BA": x_ba.tolist(),
        "trained_material_accuracy": trained_accuracy,
        "frozen_plasticity_control_accuracy": frozen_accuracy,
        "restored_virgin_material_control_accuracy": restored_accuracy,
        "mean_signed_probe_margin": float(np.mean(signed_margins)),
        "std_signed_probe_margin": float(np.std(signed_margins)),
        "interpretation": (
            "After all fast activity and eligibility are erased, an identical neutral "
            "probe recovers AB versus BA from the learned material. The mirror learned "
            "operators have the same eigenvalues, so the remembered distinction lives "
            "in input/operator/readout geometry, not in a changed eigenvalue spectrum."
        ),
    }

    # Receipts should fail loudly if the mechanism disappears.
    assert eig_max_difference < 1e-12
    assert np.max(np.abs(eig_ab)) < 1.0
    assert float(y_ab[1] - y_ab[0]) > 0.0
    assert float(y_ba[1] - y_ba[0]) < 0.0
    assert trained_accuracy > 0.90
    assert 0.47 < frozen_accuracy < 0.53
    assert 0.47 < restored_accuracy < 0.53

    out = Path("results") / "gate3_material_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
