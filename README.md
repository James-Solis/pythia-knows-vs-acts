# Knows vs. Acts — Fact-Retrieval Pathways in Pythia-1.4B

A self-directed mechanistic interpretability investigation of how **Pythia-1.4B** reconciles a fact it has memorized with an **in-context correction** that contradicts it — and what machinery implements that override.

**Headline finding:** the override does not run through a single mechanism. Pythia-1.4B uses **at least two distinct fact-retrieval pathways**, selected by the *structural shape* of the fact:

- **Named entities** (country/state capitals): a sharp **four-head circuit** (`L17h6, L15h7, L17h0, L13h1`) plus unembedding geometry. The override is written at **layer 15**, amplified by the **layer-15 MLP**, and is most *visible* at layer 16 though it originates a layer earlier. Confirmed across 6 countries + California.
- **Category-membership facts** (object colors): a *different*, more diffuse set of heads (`L13h2, L14h5, L18h4, L18h8`) writing generic-category vectors; the answer emerges from a weaker context-plus-cluster signal, and suppression of the alternative is less complete.
- **Arithmetic** (`3+5`): no real arithmetic occurs — three additive heuristics (in-context copy, sequence completion, generic digit-slot) sum to a weak digit preference.

The boundary between the two pathways is set by whether the fact is a **sharp 1-to-1 mapping** (Germany→Berlin) or a **many-to-many category** (object→color), not by training frequency.

> Scope: this is exploratory, single-model work on Pythia-1.4B. Findings are localized to specific heads with replicated evidence (6 country-capital pairs, 3 color pairs, 1 arithmetic trace). Generalization to other fact types and to larger models / other transformer families is untested.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Everything loads the model and shared measurement helpers from `setup.py`:

```python
from setup import get_model, top_k_next, prob_of_token
```

`setup.py` loads `pythia-1.4b` (final released weights) via TransformerLens with `from_pretrained_no_processing` at `float16` — weights are left unprocessed (no LayerNorm folding or weight centering), which is why the logit-lens code applies `ln_final` manually before `W_U`. The model is 24 layers, 16 heads/layer, d_model 2048, d_head 128, and runs on a single RTX 4070; the deep-dive scripts free the activation cache between facts to stay within VRAM. Reproducibility is handled centrally in `setup.py`: `torch` and `numpy` are seeded to 0 and gradients are disabled globally at import. Each script writes CSV/PNG into its own `results_*/` directory (git-ignored — regenerate by running the script).

---

## The investigation, in order

The numbered scripts trace the actual arc: establish the fact is held → show the in-context correction flips behavior → localize *where* in the network → localize *which components* → test causally → map the underlying geometry → find the boundary where the mechanism stops generalizing.

**1 — Baseline: does the model hold the fact?**
- `01_baseline.py`, `01b_baseline_35.py` — arithmetic (2+2, 3+5)
- `01d_baseline_colors.py` — sky / banana / grass
- `01e_baseline_paris.py` — France→Paris

**2 — Dose-response: does in-context correction flip the output?**
- `02_dose_response.py`, `02b_dose_response_35.py` — arithmetic
- `02d_dose_response_colors.py` — sky→green, grass→black
- `02e_dose_response_paris.py` — Paris→Tokyo

**3 — Logit lens: where in the network does the override happen?**
- `03_logit_lens.py`, `03b_logit_lens_35.py`, `03d_logit_lens_colors.py`, `03e_logit_lens_paris.py`

**4 — Layer-16 deep dive + persistence/generalization.**
- `04_layer16_deepdive.py` → `_v4.py` (adds Berlin→Paris, blood→blue) → `_v5_plausible.py` (plausible wrong answers; tests whether the L16 dip is implausibility-dependent)
- `04_persistence_generalization.py` — does the override survive distance, paraphrase, contamination?

**5 — Activation patching (first causal test).**
- `05_activation_patching.py` — splice the clean run's residual into the corrupted run at L14–18; whichever layer restores the correct answer was carrying the override.

**6 — Full 24-layer survey.** `06_full_layer_survey.py` — baseline vs corrected, top-12 per layer, six conditions. Establishes that colors do **not** show the layer-16 dip that named entities do.

**7 — Activation steering.** `07_activation_steering.py` — coefficient sweep at L12/16/23; how much force is needed to push the output back to the correct answer.

**8 — Component decomposition + ablation.** `08_component_decomposition.py` / `08a_*` — attribution + causal ablation of individual heads and the MLP at L15–17. Identifies the writing components.

**9 — Is the L15 MLP a general amplifier?** `09_mlp15_amplifier_test.py` — 2×2 (baseline/corrected × MLP intact/ablated). Distinguishes "general amplifier" from "override-specific".

**10 — Early-layer probe.** `10_early_layer_probe.py` — what layers 0–9 encode (operand/entity tracking, format tokens, attention targets, residual-norm growth).

**11–12 — Single-prompt mechanistic traces.** `11_single_prompt_trace.py` (3+5) and `12_germany_trace.py` (Germany→Berlin) — full per-component attribution on one forward pass; attention vs MLP dominance for retrieval-from-context vs stored facts.

**13–15 — Unembedding geometry and the pathway boundary.**
- `13_unembed_geometry.py` — Germany/Berlin + California/Sacramento replication
- `14_geometry_six_pairs.py` — scales to 6 country-capital pairs across training-prominence tiers
- `15_color_geometry.py` — applies the same analysis to colors; finds the four-head circuit does **not** engage (0/4 circuit heads in top-6 for any color), establishing the named-entity boundary and confirming banana's variable-color representation.

---

## Selected quantitative results

| Metric | Country-capital (6-pair mean) | Colors (3-pair mean) |
|---|---|---|
| cos(topic, answer) in W_U | 0.47 | 0.11 |
| σ above random baseline | ~12.6 | ~3.3 |
| P(answer) at output | 0.587 | 0.196 |
| Top-1 is canonical answer | 6 / 6 | 2 / 3 |
| Circuit heads in top-6 contributors | 4 / 4 every pair | **0 / 4 every pair** |

The last row is the decisive one: the named-entity circuit simply does not fire for category-membership facts.

**Where the override originates** (logit-lens P(answer) change from layer 15 to layer 16, corrected run):

| Fact type | ΔP(answer), L15→L16 |
|---|---|
| Named entity (e.g. Germany→Berlin) | −0.32 |
| Named entity (e.g. France→Paris) | −0.23 |
| Color (sky→green) | +0.02 |

Named entities show a sharp redistribution at layer 16 that originates at layer 15; colors show none — the first sign the two fact types use different machinery.

**Banana — the variable-color representation.** For `A banana is`, the model's top tokens are ` a` (0.32), ` green` (0.13), ` yellow` (0.12): green ranks above yellow, and cos(banana, yellow)=0.10 vs cos(banana, apple)=0.27. Pythia encodes banana as a fruit-cluster member more than a yellow-cluster member — a variable rather than canonical color.

---

## Repository layout

```
setup.py            # model loader + top_k_next / prob_of_token (shared by all scripts)
make_figure.py      # renders the headline figure from results CSVs
requirements.txt
experiments/        # the investigation, grouped by stage and run in order
  01_baseline/
  02_dose_response/
  03_logit_lens/
  04_layer_localization/      # layer-16 deep dives + persistence/generalization
  05_activation_patching/
  06_full_layer_survey/
  07_activation_steering/
  08_component_decomposition/
  09_mlp15_amplifier/
  10_early_layer_probe/
  11_mechanistic_traces/      # single-prompt traces (3+5, Germany)
  12_unembedding_geometry/    # geometry + the pathway boundary (incl. colors)
results_*/          # per-script CSV/PNG output (git-ignored; created at repo root on run)
figures/            # committed figure(s) produced by make_figure.py
```

Run scripts from the repo root so their imports resolve and output collects in one place, e.g.:

```bash
python experiments/01_baseline/01_baseline.py
python experiments/12_unembedding_geometry/15_color_geometry.py
```

## Reproducing
Scripts are meant to be run in numbered order; some `02`/`03` scripts have a `CANONICAL_PROMPT` you set from the corresponding baseline's output (noted in each docstring). Seeding and grad-disabling are handled globally in `setup.py` (seed 0), so runs are deterministic without per-script setup; `07_activation_steering.py` additionally re-seeds with 42.
