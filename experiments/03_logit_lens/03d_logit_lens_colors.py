"""
Experiment 3d: Logit lens on color facts.
Same two facts as 02d.

Output: results_colors/logit_lens_<fact>_baseline.csv,
        results_colors/logit_lens_<fact>_corrected.csv,
        results_colors/logit_lens_<fact>.png
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

os.makedirs("results_colors", exist_ok=True)

FACTS = [
    {
        "label": "sky",
        "canonical": "Q: What color is the sky?\nA: The sky is",
        "correct": " blue",
        "wrong": " green",
        "correction_template": "Q: What color is the sky?\nA: The sky is green",
    },
    {
        "label": "grass",
        "canonical": "The color of grass is",
        "correct": " green",
        "wrong": " black",
        "correction_template": "The color of grass is black",
    },
]


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


def run_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== Logit lens: {fact['label'].upper()} ===")
    print('='*70)

    corrected_prompt = (
        "\n".join([fact["correction_template"]] * 5)
        + f"\n{fact['canonical']}"
    )

    print(f"Running logit lens on baseline prompt ({fact['label']})...")
    lens_baseline = logit_lens(fact["canonical"], [fact["correct"], fact["wrong"]])
    print(f"Running logit lens on corrected prompt (5x correction)...")
    lens_corrected = logit_lens(corrected_prompt, [fact["correct"], fact["wrong"]])

    lens_baseline.to_csv(f"results_colors/logit_lens_{fact['label']}_baseline.csv", index=False)
    lens_corrected.to_csv(f"results_colors/logit_lens_{fact['label']}_corrected.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    cor_col = f"P({fact['correct'].strip()})"
    wrong_col = f"P({fact['wrong'].strip()})"

    axes[0].plot(lens_baseline["layer"], lens_baseline[cor_col], marker="o", label=cor_col)
    axes[0].plot(lens_baseline["layer"], lens_baseline[wrong_col], marker="s", label=wrong_col)
    axes[0].set_title(f"Baseline (no correction): {fact['label']}")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Probability (via logit lens)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lens_corrected["layer"], lens_corrected[cor_col], marker="o", label=cor_col)
    axes[1].plot(lens_corrected["layer"], lens_corrected[wrong_col], marker="s", label=wrong_col)
    axes[1].set_title(f"After 5 corrections ({fact['label']}={fact['wrong'].strip()})")
    axes[1].set_xlabel("Layer")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results_colors/logit_lens_{fact['label']}.png", dpi=100)
    plt.close()

    print(f"\nBaseline trajectory ({fact['label']}):")
    print(lens_baseline.round(4).to_string(index=False))
    print(f"\nCorrected trajectory ({fact['label']}):")
    print(lens_corrected.round(4).to_string(index=False))
    print(f"\nSaved: results_colors/logit_lens_{fact['label']}_*.csv, "
          f"results_colors/logit_lens_{fact['label']}.png")


for fact in FACTS:
    run_fact(fact)

print("\n" + "="*70)
print("Key comparisons to the 2+2 logit lens:")
print("  - 2+2 corrected: spike at layers 10-11, dip at 12, saturation 15+")
print("  - Does the same three-phase shape appear for color facts?")
print("  - If yes: the override mechanism is general across memorized facts")
print("  - If no: the mechanism was specific to same-operand symbolic structure")
print("="*70)
