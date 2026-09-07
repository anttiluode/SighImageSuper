# SighImageSuper

![pic](pic.png)

## What survives when a system is fed its own tail?

This repo starts from a very small image experiment and treats it as an operator laboratory.

Take an image state `x`, apply a spatial operator `F`, feed the result back as the next input, and repeat:

```text
x_(n+1) = alpha * x_original + (1-alpha) * F(x_n)
```

At `alpha = 0` the loop is closed. The system sees only descendants of its own previous output.

The first striking observation was visual: under the high-pass preset, a photograph turns into a moving/fading spectral cloud and then locks onto a checkerboard. The raw amplitude becomes tiny, but the normalized displayed shape becomes extremely pure.

That is not magic and it is not yet AI model collapse. It is a clean view of repeated operator dynamics.

## The important correction

For the current 64x64 Fourier operator, the high-pass preset ends at gain `1.0`. The normalized radial `k^2` lookup gives that exact gain only to the maximum-frequency bin. On an even 64x64 grid that bin is the joint Nyquist corner `(-32,-32)`, whose real spatial pattern is a checkerboard.

So this case is stronger than "the least-dying mode wins":

> **the checkerboard is an invariant mode of the operator.**

After all modes with gain `< 1` disappear, the loop is left with the original image's projection onto that one preserved mode. Its amplitude need not continue toward zero; it can plateau at the tiny coefficient that the original image happened to contain in that direction.

For a general linear operator `A`,

```text
x_n = A^n x_0 = sum_i c_i lambda_i^n phi_i
```

and the loop reveals the modes with the largest `|lambda|`.

- `|lambda| < 1`: memory decays.
- `|lambda| ~= 1`: long-lived memory.
- `|lambda| = 1`: invariant memory.
- `|lambda| > 1`: amplification / instability.
- negative or complex lambda: alternation / rotation / oscillation.

A useful way to say this is:

> **A structure does not only transform signals. It assigns them forgetting times.**

For a discrete mode with `0 < |lambda| < 1`, its half-life in recursive steps is

```text
n_half = log(1/2) / log(|lambda|)
```

That turns the operator spectrum into a literal memory-timescale spectrum.

## Why this belongs with the earlier repos

The recurring line has been:

> **structure compiles an operator.**

SighImageSuper adds a missing consequence:

> **the operator compiles a hierarchy of persistence.**

That connects several older experiments without pretending they are the same system.

### Operaattori / GeometricNeuron

Morphology determines a cable operator. A perturbation can be decomposed into spatial modes with different decay times. Fast components disappear first; slow modes dominate the late state. Geometry therefore chooses not only where a signal travels but which components of history remain available later.

### JelloBrain

JelloBrain made the slow material into the operator: signals make roads and roads alter later signals. The recursive image loop is the frozen-operator control case. The more interesting next case couples state and operator:

```text
x_(t+1)     = A(theta_t) x_t + input_t
theta_(t+1) = G(theta_t, x_t, consequence_t)
```

Now a mode that survives longer can drive more plasticity, and that plasticity can make the same mode survive even longer. That is a route to **self-canalization**: a rich-get-richer loop between state and structure.

JelloBrain's failure-gated plasticity result suggests the counter-mechanism: **a stable thing should stop teaching itself merely because it is stable.**

### 368 / fast forgetting + retained memory

The same spectrum gives a physical version of fast/slow memory. Modes far inside the unit circle forget quickly. Modes near the unit circle act as retained traces. A separate memory database is not the only way a system can retain history; persistence can be embedded in the dynamics of the same substrate being read.

Fresh-data `alpha` is then not a cosmetic control. For linear `A`, the forced fixed point is

```text
x* = alpha [I - (1-alpha) A]^-1 x_original
```

when the inverse exists. At `alpha=0`, recursive inheritance selects the operator's persistent modes. At `alpha>0`, the outside world continually re-enters and the result is a weighted mixture rather than pure self-consumption.

### AI model collapse

The current Sigh loop is **signal collapse under a fixed operator**, not full model collapse: the operator itself is not being retrained.

The closer AI analogue begins when outputs alter the operator that produces future outputs, or when training data becomes increasingly self-generated:

```text
state -> model output -> training signal -> changed model -> next output
```

Then a small representational preference can become self-reinforcing. Sigh gives us an unusually visible baseline for that process because we can first understand the fixed operator exactly and only then allow the operator to move.

## Dendrite question

A passive dendrite is not literally this FFT filter, but the mathematical family resemblance is real. Over a finite time step its cable dynamics act like an evolution operator. The voltage pattern can be decomposed into modes with different time constants; late activity is biased toward the modes that morphology and boundary conditions preserve longest.

The safe hypothesis is therefore not:

> the neuron stops because it found its eigenmode.

It is:

> **a dendritic tree may erase most components of a perturbation faster than others, leaving later readout dominated by a morphology-selected low-dimensional set of modes.**

Active conductances, NMDA, inhibition and plasticity make the operator state-dependent. At that point ordinary eigenvectors may no longer be enough; finite-time singular vectors, Jacobians, Lyapunov directions and attractors become the right objects.

That is where this stops being a textbook power-iteration demo and becomes relevant to the rest of the research program.

## Gates

### Gate 0 — analytical receipt

Reproduce the checkerboard endpoint from the exact 64x64 high-pass operator. Predict the surviving Fourier bin before iterating, then verify the 1024-step state against the analytical projection.

Run:

```bash
python gate0_spectral_receipt.py
```

### Gate 1 — fresh-data phase diagram

Sweep `alpha` and measure raw energy, spectral entropy, effective mode count and distance from the closed-loop invariant subspace. Verify the linear fixed-point formula.

### Gate 2 — moving operator

Let the EQ gains adapt slowly from the recursive state. Compare:

```text
Hebbian / reinforce-what-survives
anti-collapse / flatten-dominance
failure-gated / change-only-when-needed
```

The question becomes whether recursive state can sculpt an operator that then preferentially regenerates the same state.

### Gate 3 — dendritic operator

Use an Operaattori morphology as the operator. Excite thousands of random compartment patterns, propagate them through passive cable dynamics, and measure how effective rank and mode energy change with delay. Shuffle morphology as the attacker.

### Gate 4 — active dendrite

Add local nonlinear conductances / NMDA. Ask whether active dynamics preserve naturally slow modes, rescue normally fast-decaying directions, or create new state-dependent attractors.

### Gate 5 — AI operator microscope

Replace the image FFT operator with a denoiser, autoencoder, recurrent hidden-state update, residual-block Jacobian, or other dimension-preserving learned map. Feed its output back and measure fixed points, cycles, transient amplification and mode collapse.

For non-normal operators, do not assume the important direction is an eigenvector. Measure singular-value growth and transient amplification too.

## Claim boundary

Power iteration, eigenmodes, cable modes, dynamical systems, model collapse and continual-learning memory are established subjects. This repository does not claim to have invented them or to have discovered a new biological mechanism.

The useful research program is narrower:

> **build one visual instrument that asks the same operational question of image filters, dendrites and learned systems: when a state is repeatedly transformed by the structure that contains it, what survives, what dies, and what happens when the surviving state is allowed to rewrite the structure?**
