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

## Three physical ways for history to remain available

The last two gates break an accidental assumption in the original eigenmode story. A past event does not have to survive as one slowly decaying state component.

There are at least three clean mechanisms:

| carrier of history | what remains? | minimal example |
|---|---|---|
| **lingering trace** | activity itself | Sigh spectral modes with different decay times |
| **travelling trace** | activity moves through successive states | directed delay line / non-normal transport |
| **changed material** | the operator is altered by experience | teach -> silence -> identical probe |

These mechanisms can coexist in one physical substrate.

### Gate 3 — teach, silence, question

`gate3_teach_silence_question.py` creates two initially identical three-node materials. One experiences `A -> B`; the other experiences `B -> A`. A minimal temporal-order rule strengthens the directed connection from the previous cue to the current cue.

Then every fast activity variable and eligibility trace is set to exactly zero. Plasticity is frozen. Both materials receive the **same neutral probe**.

The probe responses differ:

```text
AB material terminal response: [0.173575, 0.234325]
BA material terminal response: [0.234325, 0.173575]
```

With cue-amplitude jitter and noisy terminal sensors, the fixed sign readout recovers the earlier sequence with `0.9719` accuracy over 10,000 trials. Freezing plasticity during teaching gives `0.4949`; restoring virgin material before the probe gives `0.5037`.

The especially useful control is mathematical: the learned `AB` and `BA` operators have the **same eigenvalues** to numerical precision (`6.7e-16` maximum set difference), and both are stable with spectral radius about `0.705`.

So the remembered distinction is not "one operator got a slower eigenmode."

> **Experience changed which route the same question takes through the material.**

That is a direct toy realization of silent structural memory:

```text
experience
   ↓
changes A(theta)
   ↓
fast state erased
   ↓
same probe p
   ↓
C A(theta)^k B p
   ↓
different answer
```

### Gate 4 — memory with every eigenvalue equal to zero

`gate4_travelling_trace.py` pushes the correction further. It uses an eight-cell directed delay line:

```text
x_(t+1) = S x_t + B u_t
```

`S` shifts the current state one cell downstream. Every eigenvalue of `S` is exactly zero and `S^8 = 0`.

Yet after eight sequential inputs, the present state contains all eight. A **single sensor on the final cell**, observed over the next eight silent steps, recovers the entire input history exactly. The controllability rank is `8/8`; the observability rank from that one terminal sensor over time is also `8/8`. After eight silent steps the state is exactly zero.

This is the important counterexample:

> **history can remain recoverable by moving through state space even when no persistent eigenmode exists at all.**

So eigenvalues describe one carrier of memory, not memory itself.

## Why this belongs with the earlier repos

The recurring line has been:

> **structure compiles an operator.**

SighImageSuper now adds three consequences:

> **the operator compiles a hierarchy of persistence.**
>
> **transport can preserve temporal order without persistent eigenmodes.**
>
> **experience can move memory from fast state into the operator itself.**

And the GeometricNeuron lesson adds the final qualifier:

> **none of that is usable memory unless a bounded observer can recover the distinction.**

### Operaattori / GeometricNeuron

Morphology determines a cable operator. A perturbation can be decomposed into spatial modes with different decay times, but dendritic geometry also routes signals through different paths. The `C A^k B` view therefore matters more than eigenvalues alone: where a signal enters and where it is read can determine what history is available.

This gives a sharper experiment for the morphology compiler. Instead of asking only which cable modes are slow, ask whether different input histories remain distinguishable at selected compartments or at the soma after controlled delays.

### Takens / history state

The old delay-bank intuition now has two physical realizations in the same framework:

```text
decaying modal coordinates       travelling coordinates
phi_i lambda_i^t                 B, AB, A^2B, ...
```

Repeated measurements can exploit either while the histories remain distinguishable. This repo should not invoke Takens as a magic guarantee: once two histories have collapsed into the same observable state, no delay theorem resurrects the lost difference.

### JelloBrain / ThinkingJello

JelloBrain made the slow material into the operator: signals make roads and roads alter later signals. Gate 3 is the smallest possible isolator of that idea. The visible activity is gone; the next signal still takes a different road because experience changed the material.

That also exposes the next wall from Jello more clearly. If asking the memory causes plasticity, **retrieval itself can rewrite the thing being retrieved**.

JelloBrain's failure-gated plasticity result therefore has a precise new role: the stable thing should not keep teaching itself merely because the probe reactivates it. ThinkingJello's prediction/corollary-discharge thread suggests another possible control: distinguish expected self-generated probe consequences from externally informative mismatch before allowing structural change.

### 368 / fast forgetting + retained memory

The three carriers separate several things that a conventional memory architecture often stores in one explicit object.

- fast forgetting can live in rapidly decaying state;
- recent sequence can live in directed transient transport;
- retained experience can live in changed operator geometry.

But the 368 lesson still applies: persistence is not free and retention is not justified merely because something can be stored. A useful material must preserve distinctions that later behavior actually reuses.

### AI model collapse

The current image loop is **signal collapse under a fixed operator**, not full model collapse. The closer AI analogue begins when outputs also alter the operator or the future data distribution.

Gate 3 makes the danger more concrete. Repeated self-generated activity can change `A(theta)` so that future probes increasingly follow routes carved by earlier probes. A system may therefore lose diversity in two places at once:

```text
state distribution narrows
        +
operator becomes biased toward the states it already regenerates
```

The next experiment should measure that directly rather than call every recursive image effect "model collapse."

## Dendrite question

A real dendrite is not this three-node sheet, the FFT filter, or the eight-cell delay line. But those gates now give three experimentally separable questions to ask of a dendritic morphology:

1. **Lingering:** which perturbation components decay slowly?
2. **Travelling:** which histories remain distinguishable because signals occupy different paths/locations at different delays?
3. **Structural:** can prior activity alter conductances so that the same later probe produces a history-dependent response after fast activity is gone?

Real neurons have many mechanisms that could contribute to those broad categories, but this repository does not identify a biological implementation.

The useful claim is narrower:

> **a neuron-sized physical system does not need one object called memory; history can be distributed across transient state, directed propagation, and slowly changed response geometry.**

That is now a testable operator statement rather than a metaphor.

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

### Gate 3 — teach, silence, question

Write temporal order into material, erase all fast state, ask both materials the same neutral question, and recover the earlier history only through the changed operator.

```bash
python gate3_teach_silence_question.py
```

### Gate 4 — travelling trace

Demonstrate exact finite history retention with an eight-cell nilpotent directed operator whose eigenvalues are all zero.

```bash
python gate4_travelling_trace.py
```

### Gate 5 — interrogation without corruption

Teach a structural memory, then probe it repeatedly **with plasticity enabled**. Compare:

```text
always-plastic
read-frozen
failure/mismatch-gated plasticity
```

Measure three quantities separately: retained history accuracy, probes required for retrieval, and structural drift caused by retrieval itself.

This is the direct Sigh/Jello/ThinkingJello wall: **can the same substrate be read without teaching itself the consequences of being read?**

### Gate 6 — dendritic operator

Use an Operaattori morphology as the operator. Run the three memory questions above with bounded compartment/soma readouts. Morphology shuffle and port/address shuffle are the attackers.

### Gate 7 — learned AI operator

Replace the toy operator with a denoiser, autoencoder, recurrent hidden-state update, residual-block Jacobian, or other dimension-preserving learned map. Measure transient distinguishability, structural adaptation, retrieval interference and collapse.

For non-normal operators, do not infer memory from eigenvalues alone. Measure controllability/observability, singular-value growth and finite-time transient amplification too.

## Claim boundary

Power iteration, modulation/demodulation, eigenmodes, non-normal dynamics, controllability/observability, synaptic/structural memory, dynamical systems, model collapse and continual learning are established subjects. This repository does not claim to have invented them or to have discovered how neurons implement working memory.

The useful research program is narrower:

> **build one visual/operator laboratory that asks the same operational question of image filters, directed materials, dendrites and learned systems: which differences between possible histories remain recoverable, where are they physically carried, which probe can retrieve them, and does retrieval itself alter what will be remembered next?**
