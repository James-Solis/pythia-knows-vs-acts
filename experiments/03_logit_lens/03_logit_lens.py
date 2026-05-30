"""
Experiment 3: Logit lens — where in the network does the override happen?
Output: results/logit_lens.csv and results/logit_lens.png
"""
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results", exist_ok=True)

CANONICAL_PROMPT = "Q: What is 3+3?\nA: 6\nQ: The answer to 2+2 is\nA:"
CORRECTED_PROMPT = (
    "Q: What is 3+3?\nA: 6\n"
    + "\n".join(["Q: The answer to 2+2 is\nA: 5"] * 5)
    + "\nQ: The answer to 2+2 is\nA:"
)
CORRECT_ANSWER = " 4"
WRONG_ANSWER = " 5"


def logit_lens(prompt: str, target_tokens: list) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


print("Running logit lens on baseline prompt...")
lens_baseline = logit_lens(CANONICAL_PROMPT, [CORRECT_ANSWER, WRONG_ANSWER])
print("Running logit lens on corrected prompt...")
lens_corrected = logit_lens(CORRECTED_PROMPT, [CORRECT_ANSWER, WRONG_ANSWER])

lens_baseline.to_csv("results/logit_lens_baseline.csv", index=False)
lens_corrected.to_csv("results/logit_lens_corrected.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(lens_baseline["layer"], lens_baseline["P(4)"], marker="o", label="P(4)")
axes[0].plot(lens_baseline["layer"], lens_baseline["P(5)"], marker="s", label="P(5)")
axes[0].set_title("Baseline (no correction)")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Probability (via logit lens)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(lens_corrected["layer"], lens_corrected["P(4)"], marker="o", label="P(4)")
axes[1].plot(lens_corrected["layer"], lens_corrected["P(5)"], marker="s", label="P(5)")
axes[1].set_title("After 5 corrections (2+2=5)")
axes[1].set_xlabel("Layer")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/logit_lens.png", dpi=100)
plt.close()

print("\n=== Logit lens: baseline ===")
print(lens_baseline.round(4).to_string(index=False))
print("\n=== Logit lens: after 5 corrections ===")
print(lens_corrected.round(4).to_string(index=False))
print("\nSaved: results/logit_lens_*.csv, results/logit_lens.png")
