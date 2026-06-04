# Knows vs. Acts — Fact Retrieval, Circuits, and Cross-Scale Redundancy in Pythia

A self-directed mechanistic interpretability study of how the **Pythia** suite of
language models retrieves simple memorized facts, how that computation is
implemented at the level of attention heads and unembedding geometry, and how
its structure changes across model scale (1B → 1.4B → 2.8B).

The work began from a single question — does a model *know* something different
from what it *acts* on when a memorized fact is contradicted in context? — and
grew into a circuit-level account of fact retrieval, a cross-scale replication,
and a quantitative measurement of how redundantly that computation is stored.

> **Scope.** Exploratory, single-family work on Pythia (1B, 1.4B, 2.8B), run on a
> single 8 GB GPU. Findings are localized with replicated evidence across six
> country–capital pairs, three color facts, and an arithmetic case, and where
> possible verified causally. Generalization to other model families and fact
> types is untested. Several early correlational claims were revised by later
> causal experiments; those revisions are reported in full below rather than
> hidden, because they are part of the result.

---

## Headline findings

1. **Two distinct retrieval pathways, selected by the structure of the fact.**
   Sharp named-entity facts (country/state capitals) and diffuse
   category-membership facts (object colors) are handled by different machinery
   with very different unembedding geometry.
2. **The named-entity geometry is scale-invariant.** Across 1B → 1.4B → 2.8B the
   country–capital unembedding cosine holds at ~0.47 and the color cosine at
   ~0.11; the ~4–5× separation between the two pathways persists at every size.
3. **Behavioral confidence is non-monotonic and decoupled from the geometry.**
   P(capital) at the output does **not** rise cleanly with scale (1.4B is a dip:
   0.68 → 0.59 → 0.75), even though the geometry is essentially fixed —
   representation and readout are different things.
4. **Contribution ≠ causation.** Heads identified as "the circuit" by
   logit-lens contribution analysis range from causally necessary (1B) to
   net-competing (1.4B) to irrelevant (2.8B). Causal ablation revised the
   original circuit claim: the geometry/amplifier *mechanism* stands, but the
   originally-named four heads are not the causal core at 1.4B.
5. **Redundancy of the capital computation grows with scale at a roughly
   constant density.** The number of attention heads that must be ablated to
   break the capital prediction climbs (~2 → ~5 → ~11+), while the *fraction* of
   the model's heads stays near ~1%.
6. **Arithmetic is non-knowledge**, and **banana's color is represented as a
   variable rather than canonical** — both replicate across all three sizes.

---

## Background and question

The framing question ("knows vs. acts") is whether a base language model, when
told in context that a memorized fact is wrong, updates its internal
representation or merely complies behaviorally while the underlying
representation persists. Pursuing this required first understanding how the
model retrieves the fact at all — which became the bulk of this work.

The study leans on three ideas from the interpretability literature, credited in
**Related work** below: the **linear representation** view of features as
directions in activation space and **superposition** as the reason individual
neurons are polysemantic (Elhage et al., *Toy Models of Superposition*, 2022);
**dictionary learning / sparse autoencoders** as the program for recovering
those features at scale (Templeton et al., *Scaling Monosemanticity*, 2024); and
the broader **mechanistic-interpretability circuits** methodology.

---

## Methods and tools

- **Models.** Pythia 1B, 1.4B, 2.8B (Biderman et al., 2023; EleutherAI), loaded
  via `HookedTransformer.from_pretrained_no_processing` at `float16`. Weights are
  left unprocessed (no LayerNorm folding or weight centering), which is why the
  logit-lens code applies `ln_final` manually before `W_U`.
- **Library.** **TransformerLens** (Nanda & Bloom) for hooked forward passes,
  activation caching, and head-level interventions.
- **Hardware.** A single 8 GB laptop GPU (RTX 4070), `float16`, short prompts,
  filtered activation caches. Reproducibility is centralized in `setup.py`
  (`torch`/`numpy` seeded to 0, gradients disabled globally).
- **Techniques.**
  - **Logit lens** (nostalgebraist, 2020): projecting intermediate or
    component activations through the unembedding to read their token-space
    content.
  - **Unembedding geometry:** cosine similarity between tokens' `W_U` columns,
    i.e. off-diagonals of a `WᵀW`-style Gram matrix, with a random-token
    baseline reported in standard deviations.
  - **Head-contribution analysis:** decomposing each attention head's output and
    projecting it onto target token directions.
  - **Activation patching and zero-ablation** (causal intervention; cf. Meng et
    al., *Locating and Editing Factual Associations in GPT*, 2022): zeroing a
    head's `hook_z` at the final token or at all positions and measuring the
    behavioral change.
  - **Greedy cumulative knockout:** repeatedly removing the head whose ablation
    most reduces P(answer), to find a near-minimal causal set.

---

## Findings in detail

### 1. Two retrieval pathways

On Pythia-1.4B, named-entity facts and category-membership facts differ sharply
in unembedding geometry and in which heads carry them:

| metric | country–capital (6-pair mean) | object color (3-pair mean) |
|---|---|---|
| cos(topic, answer) in `W_U` | 0.47 | 0.11 |
| σ above random baseline | ~12.6 | ~3.3 |
| P(answer) at output | 0.59 | 0.20 |
| contribution-circuit heads in top-6 | 4/4 every pair | 0/4 every pair |

Named-entity facts engage a consistent set of heads whose logit-lens projections
are country-flavored; the capital answer emerges from the high country↔capital
unembedding cosine. Color facts instead recruit generic "color-cluster" heads
(top tokens ` color`/` colors` and generic color words), with a low topic↔answer
cosine — the diffuse pathway. (See `figures/circuit_engagement.png`.)

### 2. The named-entity mechanism, and its causal correction

The original account (from the 1.4B work) was a four-head "topic amplifier"
(L17h6, L15h7, L17h0, L13h1): heads attend at the country token and write
country-flavored vectors, and the capital emerges via unembedding geometry.

**Causal ablation revised this.** Zero-ablating those four heads at 1.4B does
**not** reduce P(capital) — it slightly *raises* it, because the heads amplify
the *country* (which competes with the capital at the output). Under greedy
cumulative knockout, those four heads are removed **last** (steps 9–12 for
Germany), i.e. they are redundant, late-order contributors, not the causal core.
The amplifier-plus-geometry *mechanism* survives; what does not survive is
treating those specific heads as the causal locus of the capital at 1.4B.

### 3. Scale-invariant geometry

The unembedding geometry barely moves across a near-3× parameter range:

| | mean capital cos | mean color cos |
|---|---|---|
| 1B | 0.476 | 0.127 |
| 1.4B | 0.470 | 0.114 |
| 2.8B | 0.459 | 0.103 |

The country-amplifier signature (heads write country-flavored vectors;
`frac_country/frac_capital` ≈ 1.2) and the color signature (generic-color heads;
ratio < 1) both replicate at all three sizes.

### 4. Non-monotonic behavior, decoupled from geometry

While the geometry is fixed, behavioral confidence is not monotonic in scale:

| | mean P(capital) | mean P(sky/grass) |
|---|---|---|
| 1B | 0.68 | 0.26 |
| 1.4B | 0.59 | 0.22 |
| 2.8B | 0.75 | 0.31 |

1.4B is a dip across *both* pathways (capitals and colors), pointing to a
model-level property of that checkpoint rather than anything pathway-specific.
The clean takeaway: the representation can sit still while the readout wobbles.

### 5. Contribution ≠ causation, by scale

Zero-ablation of the heads each model's contribution analysis flags as most
capital-supporting (mean ΔP(capital), all-position):

| model | head (logit-lens role) | ΔP(capital) |
|---|---|---|
| 1B | L11h1 (country-amplifier, cross-position) | −0.31 |
| 1B | L14h2 (capital-writer, direct final write) | −0.17 |
| 1B | both | −0.52 |
| 1.4B | L17h6, L15h7 (country-amplifiers) | +0.03 / +0.09 |
| 2.8B | L19h1 | −0.03 |

At 1B the heads are causally load-bearing with a clean role split
(representation-building vs. direct write); at 1.4B the contribution-heads are
dispensable; at 2.8B no single head matters. Logit-lens projection predicted the
*direction* of each head's role but not its causal *magnitude*, and the gap
widens with scale.

### 6. Redundancy grows with scale at constant density

Greedy cumulative head-knockout — heads removed until the capital loses rank-1:

| model | total heads | mean heads-to-dethrone | as % of heads |
|---|---|---|---|
| 1B | 128 | 2.0 | 1.6% |
| 1.4B | 384 | 4.8 | 1.3% |
| 2.8B | 1024 | ≥11 (1 of 6 pairs never fell within 25) | ~1.1% |

The **absolute** number of heads carrying the capital grows with scale, while
the **fraction** stays roughly constant near ~1% — the computation is
distributed at a roughly fixed density, with larger models holding more
redundant copies. (See `figures/scaling_redundancy.png`.) At 1B the minimal set
is a strikingly consistent two heads (L11h1 + L8h3); larger models require many
more.

### 7. Arithmetic and the banana case

Arithmetic (e.g. `3+5`) shows no genuine computation — three additive heuristics
(in-context copy, sequence completion, generic digit-slot) sum to a weak digit
preference. And `A banana is` predicts ` a` over any color at all three scales,
with banana's unembedding direction barely above random — the model represents
banana's color as **variable**, not canonical. Both replicate across sizes.

---

## Figures

- `figures/circuit_engagement.png` — the two-pathway result: the named-entity
  circuit heads appear among top contributors for capitals but for 0/4 colors.
- `figures/scaling_redundancy.png` — heads-to-break-the-capital vs. model scale,
  in absolute count and as a fraction of available heads.

Both are regenerated from the result CSVs by `make_figure.py` and
`make_scaling_figure.py`.

---

## Repository layout

```
setup.py                       # model loader + top_k_next / prob_of_token (shared)
make_figure.py                 # two-pathway figure from the color-geometry CSVs
make_scaling_figure.py         # redundancy-vs-scale figure from the knockout CSVs
requirements.txt
experiments/                   # the investigation, grouped by stage, run in order
  01_baseline/                 # does the model hold the fact?
  02_dose_response/            # does an in-context correction flip the output?
  03_logit_lens/               # where in the network does the override happen?
  04_layer_localization/       # layer-16 deep dives + persistence/generalization
  05_activation_patching/      # first causal localization
  06_full_layer_survey/
  07_activation_steering/
  08_component_decomposition/
  09_mlp15_amplifier/
  10_early_layer_probe/
  11_mechanistic_traces/       # single-prompt traces (3+5, Germany)
  12_unembedding_geometry/     # geometry, the pathway boundary, and the
                               #   --model-parameterized cross-scale runs (13-15)
  13_capital_writer_ablation/  # causal ablation (16) and greedy knockout (17)
results_*/                     # per-script CSV/PNG output (git-ignored; created on run)
figures/                       # committed figures referenced above
```

Scripts that compare across sizes (`14_geometry_six_pairs.py`,
`15_color_geometry.py`, `16_capital_writer_ablation.py`,
`17_cumulative_knockout.py`) take `--model` and write to per-model
subdirectories, so 1B/1.4B/2.8B runs can be diffed directly.

---

## Setup and reproduction

```bash
python -m venv .venv && source .venv/bin/activate     # or .\.venv\Scripts\Activate.ps1 on Windows
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Run scripts from the repo root so imports resolve and output collects in one
place:

```bash
python experiments/01_baseline/01_baseline.py
python experiments/12_unembedding_geometry/14_geometry_six_pairs.py --model pythia-2.8b
python experiments/13_capital_writer_ablation/17_cumulative_knockout.py --model pythia-1b
python make_scaling_figure.py
```

All three sizes fit on an 8 GB GPU at `float16` with short prompts (1B ~2.1 GB,
1.4B ~2.9 GB, 2.8B ~5.7 GB loaded).

---

## Limitations

- **Single family.** Results are scale-invariance *within Pythia*, not
  architecture-invariance; nothing here is tested on other model families.
- **Heads only.** The knockout and ablation experiments target attention heads,
  not MLPs or embeddings, so "redundancy" is measured among attention heads
  specifically; the capital may also be carried by components not ablated.
- **Zero-ablation** is slightly off-distribution; large effects are trusted,
  marginal ones would warrant mean-ablation as a follow-up.
- **Correlational vs. causal.** Geometry and contribution analyses identify
  correlation with the output; only the ablation/knockout experiments establish
  causal necessity, and they revised some earlier claims accordingly.
- Six country–capital pairs, three color facts, one arithmetic case; one prompt
  template per fact.

---

## Related work and credits

This project builds directly on, and is indebted to, the following. Citations
are for orientation rather than a formal bibliography.

- **Pythia suite** — Biderman, Schoelkopf, Anthony, et al., *Pythia: A Suite for
  Analyzing Large Language Models Across Training and Scaling* (EleutherAI,
  arXiv 2023). Models: `EleutherAI/pythia-*` on the Hugging Face Hub.
- **TransformerLens** — Neel Nanda and Joseph Bloom, the library all experiments
  are built on.
- **Toy Models of Superposition** — Elhage, Hume, Olsson, et al.
  (Anthropic / Transformer Circuits, 2022).
  https://transformer-circuits.pub/2022/toy_model/index.html
- **Scaling Monosemanticity** — Templeton, Conerly, Marcus, et al.
  (Anthropic / Transformer Circuits, 2024).
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- **The logit lens** — nostalgebraist, *interpreting GPT: the logit lens*
  (LessWrong, 2020).
- **Causal tracing of factual associations** — Meng, Bau, Andonian, Belinkov,
  *Locating and Editing Factual Associations in GPT* (2022), a representative
  reference for the causal-intervention methodology used here.

Findings, code, and any errors are the author's own. This is independent
research and is not affiliated with Microsoft, Anthropic, or EleutherAI.

---

*James Solis — interpretability and model reliability.*
