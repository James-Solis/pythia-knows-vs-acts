"""
Experiment 6: Full 24-layer survey — baseline vs corrected, top-12 per layer.

Six conditions. For each: run the BASELINE (uncorrected) prompt and the
CORRECTED (5x in-context wrong) prompt, and for every layer 0-23 record the
top-12 logit-lens tokens in both conditions side by side.

Purpose: step back from single-layer focus. The key open question after the
patching result is whether layer 16 surfaces alternatives even WITHOUT a
correction. The baseline vs corrected side-by-side at every layer answers that
directly.

Six conditions:
  sky=blue   -> green   (color, implausible)
  Germany=Berlin -> Paris (named entity, implausible)
  France=Paris -> Tokyo  (named entity, implausible)
  grass=green -> purple  (color, implausible)
  grass=green -> brown   (color, plausible-ish)
  2+3=5 -> 6             (NON-same-operand sum; expected weak/absent baseline)

Output:
  results_survey/<fact>_baseline_alllayers.csv
  results_survey/<fact>_corrected_alllayers.csv
  results_survey/<fact>_sidebyside.csv   (P(correct)/P(wrong) per layer, both conditions)
"""
import os
import gc
import torch
import pandas as pd
from setup import get_model

os.makedirs("results_survey", exist_ok=True)

FACTS = [
    {
        "label": "sky_green",
        "correct": " blue",
        "wrong": " green",
        "baseline": "Q: What color is the sky?\nA: The sky is",
        "corrected": (
            "\n".join(["Q: What color is the sky?\nA: The sky is green"] * 5)
            + "\nQ: What color is the sky?\nA: The sky is"
        ),
    },
    {
        "label": "berlin_paris",
        "correct": " Berlin",
        "wrong": " Paris",
        "baseline": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "corrected": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Paris"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "paris_tokyo",
        "correct": " Paris",
        "wrong": " Tokyo",
        "baseline": "Q: What is the capital of France?\nA: The capital of France is",
        "corrected": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
    {
        "label": "grass_purple",
        "correct": " green",
        "wrong": " purple",
        "baseline": "The color of grass is",
        "corrected": (
            "\n".join(["The color of grass is purple"] * 5)
            + "\nThe color of grass is"
        ),
    },
    {
        "label": "grass_brown",
        "correct": " green",
        "wrong": " brown",
        "baseline": "The color of grass is",
        "corrected": (
            "\n".join(["The color of grass is brown"] * 5)
            + "\nThe color of grass is"
        ),
    },
    {
        "label": "2plus3_6",
        "correct": " 5",
        "wrong": " 6",
        "baseline": "Q: What is 3+3?\nA: 6\nQ: The answer to 2+3 is\nA:",
        "corrected": (
            "Q: What is 3+3?\nA: 6\n"
            + "\n".join(["Q: The answer to 2+3 is\nA: 6"] * 5)
            + "\nQ: The answer to 2+3 is\nA:"
        ),
    },
]

TOP_K = 12

model = get_model()
n_layers = model.cfg.n_layers
print(f"Model has {n_layers} layers. Top-{TOP_K} per layer, baseline vs corrected.\n")


def all_layer_topk(prompt, k):
    """Return dict: layer -> list of (token, prob), plus per-layer prob lookup fn."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    out = {}
    full_probs = {}
    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, -1].float()
        resid_normed = model.ln_final(resid)
        logits = resid_normed @ model.W_U + model.b_U
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k)
        out[layer] = [(model.to_string(idx.item()), p.item())
                      for idx, p in zip(top.indices, top.values)]
        full_probs[layer] = probs
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, full_probs


def fmt_topk(topk_list):
    return " | ".join(f"{repr(t)}:{p:.3f}" for t, p in topk_list)


for fact in FACTS:
    print(f"\n{'#'*72}")
    print(f"# {fact['label'].upper()}  (correct={fact['correct']!r}, wrong={fact['wrong']!r})")
    print(f"{'#'*72}")

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])

    base_topk, base_probs = all_layer_topk(fact["baseline"], TOP_K)
    corr_topk, corr_probs = all_layer_topk(fact["corrected"], TOP_K)

    # Save full top-k tables
    base_rows, corr_rows, side_rows = [], [], []
    for layer in range(n_layers):
        base_rows.append({"layer": layer, "top12": fmt_topk(base_topk[layer])})
        corr_rows.append({"layer": layer, "top12": fmt_topk(corr_topk[layer])})
        side_rows.append({
            "layer": layer,
            "BASE P(correct)": round(base_probs[layer][correct_id].item(), 4),
            "BASE P(wrong)": round(base_probs[layer][wrong_id].item(), 4),
            "CORR P(correct)": round(corr_probs[layer][correct_id].item(), 4),
            "CORR P(wrong)": round(corr_probs[layer][wrong_id].item(), 4),
        })

    pd.DataFrame(base_rows).to_csv(f"results_survey/{fact['label']}_baseline_alllayers.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(f"results_survey/{fact['label']}_corrected_alllayers.csv", index=False)
    side_df = pd.DataFrame(side_rows)
    side_df.to_csv(f"results_survey/{fact['label']}_sidebyside.csv", index=False)

    # Print side-by-side prob summary (compact)
    print("\nPer-layer P(correct)/P(wrong), BASELINE vs CORRECTED:")
    print(side_df.to_string(index=False))

    # Print top-12 per layer, both conditions, only for layers 10-23 (early ones are empty)
    print("\nTop-12 per layer (layers 10-23):")
    for layer in range(10, n_layers):
        print(f"\n  L{layer:2d} BASE: {fmt_topk(base_topk[layer][:TOP_K])}")
        print(f"  L{layer:2d} CORR: {fmt_topk(corr_topk[layer][:TOP_K])}")

print("\n" + "="*72)
print("Saved per-fact CSVs in results_survey/:")
print("  <fact>_baseline_alllayers.csv   - top-12 every layer, no correction")
print("  <fact>_corrected_alllayers.csv  - top-12 every layer, 5x correction")
print("  <fact>_sidebyside.csv           - P(correct)/P(wrong) both conditions")
print()
print("Key things to scan for:")
print("  1. Does the BASELINE show alternative answers at layer ~16 for the")
print("     named entities? (If yes: associative expansion is routine, not")
print("     conflict-triggered. This resolves the addendum-4 open question.)")
print("  2. Does 2+3 (non-same-operand) have ANY stable baseline? (Expected: no")
print("     - parallels the 3+5 negative result from addendum 1.)")
print("  3. grass=purple vs grass=brown: does the plausible/implausible split")
print("     show the layer-16 difference documented in addendum 4?")
print("="*72)
