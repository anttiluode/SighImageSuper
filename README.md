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

## The checkerboard correction

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

## The cloud is the more interesting part

The endpoint checkerboard is simple. The strange-looking cloud before it is where many modes are still alive.

There is a direct way to expose what that cloud is doing. Define a checkerboard modulation operator

```text
M(i,j) = (-1)^(i+j)
```

and demodulate the recursive state:

```text
e_t = M x_t
```

Because `M^-1 = M`, the envelope evolves as

```text
e_(t+1) = M A M e_t
```

This is a similarity transform: it does not change the operator eigenvalues. It changes the coordinates in which we look at the same dynamics.

For an even grid, multiplying by the checkerboard shifts the joint Nyquist corner exactly to DC. The mode that looks like "maximum pixel detail" in the original image coordinates becomes the constant mode in envelope coordinates.

So the pre-checkerboard state can be understood as a **slow spatial envelope riding on a fast Nyquist checkerboard carrier**. Recursive high-pass selection in pixel coordinates becomes low-pass-like relaxation of that envelope toward a constant field.

This is an important warning as well as an explanation:

> **persistence is an operator property; visual complexity is basis-dependent.**

The system does not preserve the checkerboard because "detail itself is memory." It preserves one particular invariant direction, which happens to look like maximum-frequency alternation in the pixel basis.

`gate1_nyquist_envelope.py` checks the demodulation identity exactly and watches the envelope collapse onto DC. In the deterministic receipt, envelope DC power goes from about `0.00038` initially to `0.819` by step 16, `0.99953` by step 64, and `1.0` by step 1024.

## The transient may be the actual memory reservoir

The pure checkerboard is a one-dimensional endpoint. If brain/AI relevance exists here, it is more likely to live in the **transient before that endpoint**, while many distinguishable directions still survive.

Because the current Fourier operator is normal and diagonal in the Fourier basis, the singular values of `A^t` are known exactly: they are simply `|gain_k|^t`. That lets us ask a more useful question than "what is the final eigenmode?":

> **How quickly does a high-dimensional state map become effectively low-dimensional under recursive inheritance?**

`gate2_transient_dimension.py` measures this without depending on any particular photograph.

For the 4096-dimensional 64x64 state:

| recursive step | stable rank | effective singular modes |
|---:|---:|---:|
| 0 | 4096.0 | 4096.0 |
| 1 | 153.90 | 462.15 |
| 4 | 16.03 | 52.13 |
| 8 | 4.93 | 13.22 |
| 16 | 1.84 | 3.07 |
| 32 | 1.10 | 1.21 |
| 64 | 1.002 | 1.005 |

The map remains algebraically full-rank because every gain is positive, but under any finite precision/noise floor almost all directions rapidly become unusable.

That suggests a stronger research object:

> **memory is not just what survives; it is what differences between possible histories remain distinguishable to a later readout.**

For a linear input/state/readout system, the relevant family is

```text
K_k = C A^k B
```

where `B` says where an event enters, `A` says how the substrate evolves, and `C` says what the bounded observer can actually read.

That equation ties the Sigh loop much more directly to the older work than the checkerboard alone does.

## Why this belongs with the earlier repos

The recurring line has been:

> **structure compiles an operator.**

SighImageSuper adds two consequences:

> **the operator compiles a hierarchy of persistence.**
>
> **a bounded readout turns that persistence hierarchy into a hierarchy of usable memory.**

### Operaattori / GeometricNeuron

Morphology determines a cable operator. A perturbation can be decomposed into spatial modes with different decay times. Fast components disappear first; slow modes dominate the late state. Geometry therefore chooses not only where a signal travels but which components of history remain available later.

The `C A^k B` view also restores the GeometricNeuron lesson: a direction can physically survive and still be useless if the observer cannot access it.

### Takens / history state

The older delay-bank intuition does not require literal digital taps. A physical substrate can expose history through a bank of transient modes with different lifetimes. Repeated readout can exploit those changing coordinates while they remain distinguishable.

This repo should not invoke Takens as a magic guarantee: once two histories have collapsed into the same observable state, no delay theorem resurrects the lost difference. The useful quantity here is first observability/distinguishability under finite noise and only then nonlinear reconstruction.

### JelloBrain / ThinkingJello

JelloBrain made the slow material into the operator: signals make roads and roads alter later signals. The recursive image loop is the frozen-operator control case. The more interesting next case couples state and operator:

```text
x_(t+1)     = A(theta_t) x_t + input_t
theta_(t+1) = G(theta_t, x_t, consequence_t)
```

Now a mode that survives longer can drive more plasticity, and that plasticity can make the same mode survive even longer. That is a route to **self-canalization**: a rich-get-richer loop between state and structure.

JelloBrain's failure-gated plasticity result suggests the counter-mechanism: **a stable thing should stop teaching itself merely because it is stable.**

The new Sigh view makes the danger sharper. Plasticity that merely reinforces whatever is currently longest-lived may improve persistence while destroying the transient dimensionality needed to distinguish histories.

### 368 / fast forgetting + retained memory

The same spectrum gives a physical version of fast/slow memory. Modes far inside the unit circle forget quickly. Modes near the unit circle act as retained traces. A separate memory database is not the only way a system can retain history; persistence can be embedded in the dynamics of the same substrate being read.

But Gate 2 adds the missing cost: retaining one dominant mode while every other distinction dies is not rich memory. A useful memory substrate must preserve the distinctions that future behavior actually needs.

Fresh-data `alpha` is then not a cosmetic control. For linear `A`, the forced fixed point is

```text
x* = alpha [I - (1-alpha) A]^-1 x_original
```

when the inverse exists. At `alpha=0`, recursive inheritance selects the operator's persistent modes. At `alpha>0`, repeated replay of the original state continually re-enters and the result is a weighted mixture rather than pure self-consumption.

For a true grounding/model-collapse experiment, future gates should replace repeated replay of one fixed original with genuinely new external observations.

### AI model collapse

The current Sigh loop is **signal collapse under a fixed operator**, not full model collapse: the operator itself is not being retrained.

The closer AI analogue begins when outputs alter the operator that produces future outputs, or when training data becomes increasingly self-generated:

```text
state -> model output -> training signal -> changed model -> next output
```

Then a small representational preference can become self-reinforcing. The danger is not only loss of amplitude. It is loss of **distinguishable directions**: many possible histories/inputs becoming mapped into an increasingly narrow family of states.

Sigh gives us an unusually visible baseline for that process because we can first understand the fixed operator exactly and only then allow the operator to move.

## Dendrite question

A passive dendrite is not literally this FFT filter, but the mathematical family resemblance is real. Over a finite time step its cable dynamics act like an evolution operator. The voltage pattern can be decomposed into modes with different time constants; late activity is biased toward the modes that morphology and boundary conditions preserve longest.

The Nyquist demodulation gives a second useful perspective: what looked like high-frequency checkerboard selection becomes smooth envelope relaxation after a basis change. So the visual resemblance to a spreading/fading dendritic field is not evidence of identical biology, but it is less mysterious than it first appeared: both can be read as transient mixtures relaxing toward a much smaller set of slow modes.

The safe hypothesis is therefore not:

> the neuron stops because it found its eigenmode.

It is:

> **a dendritic tree may erase most components of a perturbation faster than others, leaving later readout dominated by a morphology-selected low-dimensional set of modes.**

And the stronger computational question is:

> **does dendritic geometry keep behaviorally useful histories distinguishable for the delays at which downstream machinery needs them?**

Active conductances, NMDA, inhibition and plasticity make the operator state-dependent. At that point ordinary eigenvectors may no longer be enough; finite-time singular vectors, Jacobians, Lyapunov directions and attractors become the right objects.

## Gates

### Gate 0 — analytical checkerboard receipt

Predict the exact surviving Fourier bin before iterating, then verify the 1024-step state against the analytical projection.

```bash
python gate0_spectral_receipt.py
```

### Gate 1 — Nyquist-envelope receipt

Demodulate by `(-1)^(i+j)`, prove that the Nyquist corner shifts exactly to DC, and measure the cloud/envelope collapsing toward a constant field.

```bash
python gate1_nyquist_envelope.py
```

### Gate 2 — transient operator dimension

Measure the exact singular spectrum of `A^t` and how quickly the 4096-dimensional map becomes effectively one-dimensional.

```bash
python gate2_transient_dimension.py
```

### Gate 3 — memory/distinguishability

Inject matched histories (for example `A then B` versus `B then A`) and test how long a bounded noisy readout can still distinguish them. Compare against simple decaying-trace baselines at matched state/readout cost.

### Gate 4 — moving operator

Let the operator adapt slowly from recursive state. Compare reinforcement of whatever survives, anti-collapse pressure, and failure/task-gated plasticity. The key metric is not simply persistence: does adaptation preserve useful transient distinctions?

### Gate 5 — dendritic operator

Use an Operaattori morphology as the operator. Excite many compartment patterns, propagate them through passive cable dynamics, and measure finite-time singular spectrum, effective rank and readout distinguishability with delay. Shuffle morphology as the attacker.

### Gate 6 — learned AI operator

Replace the image FFT operator with a denoiser, autoencoder, recurrent hidden-state update, residual-block Jacobian, or other dimension-preserving learned map. Feed its output back and measure fixed points, cycles, transient amplification, distinguishability loss and mode collapse.

For non-normal operators, do not assume the important direction is an eigenvector. Measure singular-value growth and transient amplification too.

## Claim boundary

Power iteration, modulation/demodulation, eigenmodes, cable modes, observability, dynamical systems, model collapse and continual-learning memory are established subjects. This repository does not claim to have invented them or to have discovered a new biological mechanism.

The useful research program is narrower:

> **build one visual instrument that asks the same operational question of image filters, dendrites and learned systems: when a state is repeatedly transformed by the structure that contains it, which differences between possible histories remain available, for how long, to which observer, and what happens when those surviving differences are allowed to rewrite the structure?**
