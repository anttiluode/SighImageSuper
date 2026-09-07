# Latest result — from recursive pixels to causal self-interrogation

The newest gates tighten two separate threads that came out of the Sigh live loop.

First, the strange visual "last gasp" around `6e-8` is a numerical effect of the original CUDA float16 storage, not a new physical/image mode. Second, the structural-memory line has now moved past the exact Gate-6 negative image into a counted learned predictor, exposed its failure, and found one active-intervention repair.

## Precision correction

The original CUDA live loop stores each recursive generation as float16. The smallest positive float16 subnormal is

```text
2^-24 = 5.960464477539063e-8
```

which matches the repeated `0.00000006` raw maximum in the recording.

`precision_floor_probe.py` uses the deliberately trivial operator

```text
A = 0.5 I
```

on a field containing exactly five real cosine modes. In exact arithmetic this operator cannot change spectral shape at all; it can only halve amplitudes.

Yet float16 storage produces:

```text
step 22: 25 relative-power Fourier bins
step 23: 251 bins
step 24: 2213 bins, max = 5.9604645e-8
step 25: exactly zero
```

while float32/float64 still have the original ten conjugate Fourier bins at step 24.

So the sparse islands / spectral spreading immediately before disappearance are a `Q16(Ax)` quantization nonlinearity. They are visually interesting but must not be interpreted as recovered information or a new eigenmode.

For a scientific rerun without changing the original experiment, use:

```bash
python sigh_image_live_loop_fp32.py
```

which launches the same GUI/operator while keeping recursive state in float32 on CUDA.

---

## Gate 7 — exact negative image is replaced by a learned local echo model

Gate 6 had exact access to the current operator and propagated a full predicted self-state. That was an ideal control, not a bounded mechanism.

Gate 7 gives each of the three nodes only an eight-lag local impulse-response template for one known self-probe:

```text
3 nodes x 8 lags = 24 predictor weights
```

Each local coefficient learns from the outgoing probe sign `a` and that node's own response:

```text
h_j[t] <- h_j[t] + lr * a * (x_j[t] - a h_j[t])
```

The predictor never reads the material matrix.

After 30 noisy calibration probes, predictor error falls from `0.2175` after one episode to `0.00131`.

With event-record plasticity then enabled for 100 self-queries:

| condition | material drift |
|---|---:|
| no predictor | `0.032836` |
| learned local predictor | `7.50e-8` |

That is about a **437,000x** drift reduction in this isolator.

The next attacker changes the valid material from the learned AB operator to the learned BA operator while leaving the old AB echo predictor stale. After 50 self-queries:

| echo-model update rate | false drift from valid BA |
|---|---:|
| frozen | `0.005016` |
| slow | `0.002196` |
| fast | `0.000119` |

So the old fast/slow theme reappears in a more precise role: a **fast self-model can re-track a legitimately changed slow material before stale self-prediction writes much false structure into it**.

But this is not solved. A persistent 10% multiplicative prediction error, with no external event at all, produces material drift `0.000286` after 100 queries. Correlated prediction error can become a false memory.

---

## Gate 8 — command correlation is not enough

A natural bounded trick is to randomize the sign of self-probes and learn the command-correlated response. This works when external disturbances are independent of the command.

After 5000 local correlation samples:

```text
external independent of command:
self-template error = 0.000387

external perfectly command-correlated:
self-template error = 0.193703
```

The corrupted predictor is disastrous in two ways.

On 100 pure self-queries it produces material drift `0.04214`, versus `1.79e-7` for the predictor trained with independent external disturbances.

And when a genuinely external event arrives with the same command correlation that the predictor learned, the corrupted negative image cancels it completely in this deterministic toy:

```text
residual with clean predictor      = 0.19409
residual with confounded predictor = 0.0
```

This is an identifiability result. If self and world vary together under every observation the system makes, correlation cannot tell them apart.

---

## Gate 9 — private dither creates causal leverage

The next move is active rather than passive.

Give the coarse self-command `a` a small private perturbation `d`:

```text
command = a + epsilon d
```

and use the balanced intervention set

```text
(a,d) = (+,+), (+,-), (-,+), (-,-)
```

with `epsilon = 0.2`.

Suppose the external event remains perfectly correlated with the coarse action but does **not** know/respond to the private dither:

```text
y = (a + epsilon d) S + a E
```

Then

```text
S_hat = (1 / 4 epsilon) sum d y = S
```

because the action-correlated external response cancels under the orthogonal dither.

The gate recovers the self echo to `1.17e-16` norm error and produces exactly zero self-query material drift in the deterministic control.

The attacker is equally informative. If the external world also responds to the private dither,

```text
y = (a + epsilon d) S + (a + kappa d) E
```

then

```text
S_hat = S + (kappa / epsilon) E.
```

For `kappa = 0.1`, the measured predictor error is `0.0968515933`, exactly the predicted `(kappa/epsilon)E` contamination to numerical precision, and 100 self-queries now drift the material by `0.01088`.

So active perturbation punches through one wall but reveals the next one:

> **A bounded system can separate causes only to the extent that its available interventions make those causes respond differently.**

That is a more useful statement than "subtract self from world."

---

## Where the repo collection now lines up

The current spine is becoming operational rather than metaphorical:

```text
EVENT
  |
  v
fast transient state
  |
  +--> lingering modal traces
  |
  +--> travelling traces
  |
  v
activity writes slow material A(theta)
  |
  v
fast state can be erased
  |
  v
active probe Bp
  |
  v
bounded readout C A(theta)^k Bp
  |
  +--> probe geometry decides what memory is visible
  |
  +--> probe activity can itself rewrite memory
  |
  v
self-model / negative image
  |
  +--> prediction error can itself become false memory
  |
  v
private causal perturbations can identify some self effects
```

This connects the earlier repos without claiming they are one biological mechanism:

- **Operaattori / GeometricNeuron:** geometry compiles transport and determines which probe/readout addresses reveal a stored distinction.
- **Takens/history:** useful history can live in finite-time trajectories, not only slow eigenmodes.
- **JelloBrain:** propagated activity changes the material; stable routes can become self-reinforcing.
- **ThinkingJello / corollary discharge:** the expected consequences of one's own intervention should not automatically become new event memory.
- **368:** fast adaptation, retained structure, retrieval, investigation and forgetting have distinct jobs; here the self-model itself benefits from being fast relative to the material it protects.
- **AlgoSchalgo / active sensing:** asking the right question can create information that passive observation cannot contain.

## Next gate

Do **not** jump to a giant 2-D sheet yet.

Gate 9 says the next clean problem is to make the observer choose its interventions causally under a probe budget while the identity of the informative branch changes across worlds.

The observer must not inspect `A(theta)`, receive the hidden AB/BA label, or be rewarded with experimenter-only conductance drift. It can learn three different things from a probe, and those learning channels must remain separate:

```text
self-echo model      <- what did my intervention cause?
query policy         <- which intervention reduces my uncertainty?
event material       <- what happened in the outside world?
```

The stronger target is no longer merely "find the differential probe."

> **Learn how to ask a changing material questions that reveal stored distinctions, while using private interventions to prevent the act of asking—and errors in predicting that act—from becoming the next memory.**

Only after that survives attackers should the exact protocol be transplanted to a local spatial sheet and then an Operaattori dendritic morphology.
