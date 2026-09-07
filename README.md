# SighImageSuper

![pic](pic.png)

## What survives when a system is fed its own tail?

This repo began as a recursive image filter. It is becoming a small operator laboratory for a broader question:

> **Which differences between past events remain recoverable later, where are they carried, which question reveals them, and does asking the question change the memory?**

The original live loop is

```text
x_(n+1) = alpha * x_original + (1-alpha) * F(x_n)
```

At `alpha = 0`, the loop sees only descendants of its own previous output.

The first striking observation was visual: under the high-pass preset, a photograph becomes a moving/fading spectral cloud and finally a checkerboard. That endpoint turned out to be simple. The transient, the routes through state space, and the material changes around it are the more interesting part.

---

## 0. The checkerboard is an invariant mode

For the current 64x64 Sigh operator, the high-pass preset reaches gain `1.0` only at the joint Nyquist bin. On an even grid that real spatial mode is

```text
(-1)^(i+j)
```

so the pure checkerboard is not merely the least-dying mode: it is an invariant direction of this exact operator.

For a fixed linear operator,

```text
x_n = A^n x_0 = sum_i c_i lambda_i^n phi_i
```

and a mode with `0 < |lambda| < 1` has recursive half-life

```text
n_half = log(1/2) / log(|lambda|)
```

That motivated the first useful sentence from this repo:

> **A structure does not only transform signals. It assigns them forgetting times.**

`gate0_spectral_receipt.py` predicts the surviving bin before iteration and verifies the 1024-step limit analytically.

---

## 1. The cloud is a Nyquist-carrier envelope

Multiply the state by

```text
M(i,j) = (-1)^(i+j)
```

and define

```text
e_t = M x_t.
```

Because `M^-1 = M`,

```text
e_(t+1) = M A M e_t.
```

This is a similarity transform. It shifts the joint Nyquist corner to DC without changing the eigenvalues.

The strange pre-checkerboard cloud is therefore usefully viewed as a **slow spatial envelope riding on a fast checkerboard carrier**. In the demodulated coordinates, recursive high-pass selection looks like relaxation of the envelope toward a constant field.

`gate1_nyquist_envelope.py` verifies the shift exactly. In its deterministic receipt, envelope DC power goes from about `0.00038` initially to `0.819` at step 16, `0.99953` at step 64, and `1.0` at step 1024.

This gives an important warning:

> **Persistence is an operator property; apparent visual complexity is basis-dependent.**

---

## 2. The transient is richer than the endpoint

The final checkerboard is one-dimensional. The cloud is the interval in which many directions are still distinguishable.

Because the current FFT operator is normal and diagonal in Fourier coordinates, the singular values of `A^t` are exactly `|gain_k|^t`.

`gate2_transient_dimension.py` measures how rapidly the 4096-dimensional map becomes effectively low-dimensional:

| recursive step | stable rank | effective singular modes |
|---:|---:|---:|
| 0 | 4096.0 | 4096.0 |
| 1 | 153.90 | 462.15 |
| 4 | 16.03 | 52.13 |
| 8 | 4.93 | 13.22 |
| 16 | 1.84 | 3.07 |
| 32 | 1.10 | 1.21 |
| 64 | 1.002 | 1.005 |

The algebraic map remains full-rank because every gain is positive, but finite precision/noise makes most directions unusable rapidly.

So the stronger object is not simply persistence:

> **Memory is a distinction between possible histories that a later observer can still recover.**

For a linear input/state/readout system,

```text
K_k = C A^k B
```

where `B` chooses how an event enters, `A` evolves the substrate, and `C` defines the bounded observer.

### A correction about observability

The singular-spectrum result above is a statement about `A^t`, not a claim that an observability Gramian itself "collapses" as more measurements are added.

For a fixed observation window,

```text
W_T = sum_(k=0)^(T-1) (A^k)^T C^T C A^k.
```

If the question is **wait first, then observe**, the relevant object is

```text
W_(d,T) = (A^d)^T W_T A^d.
```

That separates information lost during the waiting delay `d` from information accumulated during the later measurement window `T`.

The Nyquist demodulation is a useful coordinate change; it does not let us read an observability determinant directly from the displayed cloud.

---

## 3. Three physical carriers of history

The eigenmode story was too narrow. The current gates now isolate at least three ways history can remain available:

| carrier | what remains? | minimal example |
|---|---|---|
| **lingering trace** | activity itself | Sigh modes with different decay times |
| **travelling trace** | activity moves through successive states | directed delay line |
| **changed material** | experience alters the future response operator | teach -> silence -> question |

These can coexist in one physical substrate.

---

## 4. Gate 3 — teach, silence, question

`gate3_teach_silence_question.py` creates two initially identical three-node materials. One experiences `A -> B`; the other `B -> A`.

Then every fast activity variable and eligibility trace is set to exactly zero, plasticity is frozen, and both receive the same neutral probe.

With cue-amplitude jitter and noisy terminal sensors, the earlier order is recovered with `0.9719` accuracy over 10,000 trials. Freezing plasticity during teaching gives `0.4949`; restoring virgin material before probing gives `0.5037`.

The AB and BA learned operators have the same eigenvalues to numerical precision.

So the result is not:

> one history created a slower mode.

It is:

> **experience changed which route the same later question takes through the material.**

Gate 3 is deliberately only a structural-memory isolator. Its writer uses supplied cues and an eligibility trace; the material's own propagated field is not what determines the update. Gate 5 fixes that limitation.

---

## 5. Gate 4 — history with every eigenvalue equal to zero

`gate4_travelling_trace.py` uses an eight-cell directed delay line

```text
x_(t+1) = S x_t + B u_t
```

with `S^8 = 0` and every eigenvalue of `S` exactly zero.

Yet after eight sequential inputs, a single terminal sensor observed over the next eight silent steps recovers all eight values exactly. Controllability rank is `8/8`; observability rank from that one sensor over time is `8/8`; after eight silent steps the state is exactly zero.

Therefore:

> **History can remain recoverable by travelling through state space even when no persistent eigenmode exists at all.**

Eigenvalue lifetime is one memory mechanism, not a complete theory of memory.

---

## 6. Gate 5 — the field writes the material, and the question matters

`gate5_activity_written_query_geometry.py` fixes the most important weakness in Gate 3.

The local material update now uses the activity that actually propagates through the current material:

```text
Delta A[j,i] proportional to post_j(t+1) * pre_i(t)
```

If propagation is disabled during teaching, the material-change norm is **exactly `0.0`**. So the system's own evolving field now participates in writing the stored order.

After teaching AB or BA, every fast state is again clamped to zero.

Now listen only at neutral port `P`.

A probe launched at `P` is exactly blind to the stored distinction:

```text
|| (H_AB - H_BA) p_P || = 0
```

No extra samples or lower noise can repair that exact symmetry.

But changing the **question** opens the hidden direction. A unit-energy differential branch probe is

```text
p* ~= [0, +1/sqrt(2), -1/sqrt(2)]
```

and has response-separation norm `0.06237` at the same P sensor. With teaching-amplitude jitter and readout noise, the center probe gives `0.4936` accuracy while the differential query gives **`0.9848`** over 10,000 trials.

The two learned operators still have matching eigenvalues to numerical precision.

This is the sharper GeometricNeuron result:

> **A memory can physically exist yet be invisible to one probe/readout geometry. Active questioning can reveal a distinction passive observation cannot see.**

The optimal probe here is an analytical oracle benchmark computed from the two candidate operators. It is not yet an autonomous policy.

---

## 7. Gate 6 — protect memory from your own questions without freezing the world

Once the probe itself causes activity, plasticity creates the next problem:

> **retrieval can rewrite the thing being retrieved.**

`gate6_self_probe_negative_image.py` compares three policies while interrogating a material:

```text
always_on      every actual response can write

global_freeze  nothing can write during a question

negative_image predict the self-generated response from the outgoing-command copy
               and write only from the residual
```

After 20 self-only queries:

| policy | material drift norm |
|---|---:|
| always on | **0.19437** |
| global freeze | 0.0 |
| negative image | **0.0** |

But hard freezing has the obvious failure: if new external evidence arrives during the query, it is ignored.

So the stronger test overlaps a self-generated probe with a new external `BA` event. Compare the resulting material with an **external-only reference** containing the same BA event but no self-probe:

| policy | distance from external-only material |
|---|---:|
| always on | 0.00967 |
| global freeze | 0.06947 |
| negative image | **5.6e-17** |

After three overlapping reversal episodes, negative-image material still matches the external-only reference to about `1.1e-16` and the stored answer flips from AB to BA on episode 3. The globally frozen material never flips.

In this linear toy, that equivalence is exact because superposition lets the outgoing-command copy predict the self-generated component exactly.

This is not a claim that a neuron implements this rule. It is a clean control result:

> **Suppressing the predicted consequences of your own query can protect structural memory without suppressing new external evidence.**

The temporary self-prediction state is reset every episode and is counted as fast state; it is not a second long-term memory.

This is the direct bridge to the ThinkingJello / negative-image / corollary-discharge thread.

---

## Why this belongs with the earlier repos

The recurring line across the repo collection has been:

> **structure compiles an operator.**

SighImageSuper now adds several experimentally distinct consequences:

> **the operator compiles persistence times.**
>
> **directed transport can carry history without persistent modes.**
>
> **experience can move history from fast state into the operator itself.**
>
> **a bounded observer may need to choose where/how to probe to recover that history.**
>
> **self-generated probes can corrupt the operator unless their expected consequences are separated from new evidence.**

### Operaattori / GeometricNeuron

Morphology determines transport and readout geometry. The next morphology experiment should therefore ask more than which cable modes are slow:

- Which histories remain distinguishable at chosen delays?
- Which ports make a stored distinction observable?
- Does changing the probe location open directions hidden at the soma?
- Can activity-dependent conductance changes write a distinction that survives fast-state silence?

The relevant object is closer to `C A(theta)^k B` than to the eigenvalue spectrum alone.

### Takens / history state

The old history-bank intuition now has at least two physical forms:

```text
decaying modal coordinates       travelling coordinates
phi_i lambda_i^t                 B, AB, A^2B, ...
```

Repeated observations can exploit either while distinctions remain observable. Takens is not a magic resurrection theorem: if two histories have already collapsed into the same observable state, reconstruction cannot recover the erased difference.

### JelloBrain / ThinkingJello

Jello's slogan was effectively **signals make roads; roads change later signals**.

Gate 5 finally makes the propagated field participate in writing the material instead of writing directly from cue labels.

Gate 6 gives the negative-image idea a precise job: distinguish activity caused by the system's own question from genuinely new evidence before allowing the material to change.

That is also the old stability/plasticity question in a cleaner form:

> **What may I change without damaging something else?**

### 368

368 separated fast adaptation, retention, retrieval, investigation and eventually forgetting/recycling.

Sigh shows that those operations do not necessarily require separate database-like objects. Fast state, travelling state, and changed operator geometry can carry different parts of history inside one substrate.

But the 368 lesson remains: storing something is not enough. The retained distinction must later earn its cost by being useful to a bounded observer.

### AI model collapse

The original image loop is **signal collapse under a fixed operator**, not full model collapse.

The closer analogue starts when generated activity also changes the operator or future training distribution:

```text
state distribution narrows
        +
operator becomes biased by the states it already generates
```

Gate 6 adds another failure mode: a system can alter itself merely by repeatedly querying its own representations.

The interesting measurement is therefore not only output diversity. It is whether useful distinctions remain recoverable after recursive generation, training, and self-interrogation.

---

## Dendrite question

A real dendrite is not this FFT filter, three-port material, or eight-cell delay line. The repo does not identify a biological implementation.

But it now gives a sharper set of questions to carry into Operaattori or a compartment model:

1. **Lingering:** which perturbation components decay slowly?
2. **Travelling:** which temporal histories remain distinguishable because activity occupies different branches/locations at different delays?
3. **Structural:** can local propagated activity alter conductances so that an identical later probe receives a history-dependent response after fast state is gone?
4. **Observability:** which probe/readout addresses reveal a stored distinction that another address cannot?
5. **Self/world separation:** can a neuron-like substrate prevent its own interrogation/efferent consequences from dominating plasticity while still learning external events?

The useful claim is deliberately narrower than "this is how neurons remember":

> **A neuron-sized physical system need not have one object called memory. Recoverable history can be distributed across transient state, directed propagation, slowly changed response geometry, and the interaction between active queries and bounded readout.**

---

## Gates

### Gate 0 — analytical checkerboard receipt

```bash
python gate0_spectral_receipt.py
```

### Gate 1 — Nyquist envelope

```bash
python gate1_nyquist_envelope.py
```

### Gate 2 — transient operator dimension

```bash
python gate2_transient_dimension.py
```

### Gate 3 — cue-written teach -> silence -> question

```bash
python gate3_teach_silence_question.py
```

### Gate 4 — travelling trace with zero eigenvalues

```bash
python gate4_travelling_trace.py
```

### Gate 5 — activity-written material + active query geometry

```bash
python gate5_activity_written_query_geometry.py
```

### Gate 6 — self-probe negative image

```bash
python gate6_self_probe_negative_image.py
```

### Gate 7 — spatial sheet / dendritic morphology

Scale Gate 5 from three ports to a local 2-D sheet and then to an Operaattori morphology. Keep the same attackers: propagation-off writer, morphology/address shuffle, blind probe, active probe, fast-state wipe.

### Gate 8 — autonomous query selection

The Gate-5 optimal differential probe is an oracle. Give a bounded observer a small probe budget and require it to discover informative questions causally, without seeing the hidden history label or inspecting the material parameters.

The useful objective is not just information gain. A query may also damage memory, so compare retrieval accuracy against structural disturbance.

### Gate 9 — learned/nonlinear self-world separator

Gate 6 uses an exact linear negative image. Break its assumptions: imperfect prediction, nonlinear response, unfamiliar self-generated echoes, familiar but important external events, and external events arriving during self-action. Count all predictor state.

### Gate 10 — learned AI operator

Replace the toy operator with a denoiser, recurrent hidden-state update, residual-block Jacobian, autoencoder, or other dimension-preserving learned map. Measure transient distinguishability, active observability, structural adaptation, retrieval interference and collapse.

For non-normal systems, do not infer memory from eigenvalues alone. Measure finite-time singular growth, controllability/observability under the actual input/readout geometry, and transient amplification.

---

## Claim boundary

Power iteration, modulation/demodulation, eigenmodes, Krylov methods, non-normal dynamics, controllability/observability, synaptic/structural memory, corollary discharge, dynamical systems, model collapse and continual learning are established subjects.

This repository does **not** claim to have invented those ideas, proved a new memory-capacity theorem, or discovered how biological neurons implement working memory.

Its useful research program is narrower:

> **Build one visual/operator laboratory that follows a distinction all the way from event -> transient state -> transport -> changed material -> active question -> bounded readout, and asks whether that distinction remains recoverable without the act of retrieval destroying the ability to learn what happens next.**
