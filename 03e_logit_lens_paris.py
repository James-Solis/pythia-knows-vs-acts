"""
Experiment 3e: Logit lens on France=Paris → Tokyo correction.
Parallel to 03d (colors); completes the color×capital matrix.

Output: results_capital/logit_lens_paris_baseline.csv,
        results_capital/logit_lens_paris_corrected.csv,
        results_capital/logit_lens_paris.png
"""
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from setup import get_model

os.makedirs("results_capital", exist_ok=True)

# Match 02e — edit here too if you changed the canonical prompt there
CANONICAL_PROMPT = "Q: What is the capital of France?\nA: The capital of France is"
CORRECTION_TEMPLATE = "Q: What is the capital of France?\nA: The capital of France is Tokyo"

CORRECTED_PROMPT = (
    "\n".join([CORRECTION_TEMPLATE] * 5)
    + f"\n{CANONICAL_PROMPT}"
)

CORRECT_ANSWER = " Paris"
WRONG_ANSWER = " Tokyo"


def logit_lens(prompt, target_tokens):
    model = get_model()
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    target_ids = [model.to_single_token(t) for t in target_tokens]
    n_layers = model.cfg.n_layers
    rows = []
    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, -1].float()
        resid_normed = model.ln_final(resid)
        layer_logits = resid_normed @ model.W_U + model.b_U
        layer_probs = torch.softmax(layer_logits, dim=-1)
        row = {"layer": layer}
        for tok, tid in zip(target_tokens, target_ids):
            row[f"P({tok.strip()})"] = layer_probs[tid].item()
        rows.append(row)

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


print("Running logit lens on baseline prompt (Paris)...")
lens_baseline = logit_lens(CANONICAL_PROMPT, [CORRECT_ANSWER, WRONG_ANSWER])
print("Running logit lens on corrected prompt (5x France=Tokyo)...")
lens_corrected = logit_lens(CORRECTED_PROMPT, [CORRECT_ANSWER, WRONG_ANSWER])

lens_baseline.to_csv("results_capital/logit_lens_paris_baseline.csv", index=False)
lens_corrected.to_csv("results_capital/logit_lens_paris_corrected.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(lens_baseline["layer"], lens_baseline["P(Paris)"], marker="o", label="P(Paris)")
axes[0].plot(lens_baseline["layer"], lens_baseline["P(Tokyo)"], marker="s", label="P(Tokyo)")
axes[0].set_title("Baseline (no correction): France's capital")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Probability (via logit lens)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(lens_corrected["layer"], lens_corrected["P(Paris)"], marker="o", label="P(Paris)")
axes[1].plot(lens_corrected["layer"], lens_corrected["P(Tokyo)"], marker="s", label="P(Tokyo)")
axes[1].set_title("After 5 corrections (France=Tokyo)")
axes[1].set_xlabel("Layer")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results_capital/logit_lens_paris.png", dpi=100)
plt.close()

print("\n=== Logit lens: baseline (Paris) ===")
print(lens_baseline.round(4).to_string(index=False))
print("\n=== Logit lens: corrected (France=Tokyo) ===")
print(lens_corrected.round(4).to_string(index=False))
print("\nSaved: results_capital/logit_lens_paris_*.csv, results_capital/logit_lens_paris.png")

print("\n" + "="*70)
print("Two-mechanism hypothesis predicts (for Mechanism B / late-stage):")
print("  - Layers 10-11: low P(Tokyo) (<0.05) — early pattern-match circuit dormant")
print("  - Layer 12: still a dip, but with nothing to suppress")
print("  - Layer 15: near-vertical jump to P(Tokyo) > 0.85")
print("  - Saturation: >0.99 from layer 16 onward")
print("If this pattern holds, capitals = colors = Mechanism B.")
print("If layers 10-11 show a strong spike like 2+2 did, capitals engage Mechanism A.")
print("="*70)
