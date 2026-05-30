"""
Experiment 4-deep (v5): Layer 16 inspection with PLAUSIBLE wrong answers.

Parallel to v4 (which used implausible wrong answers). Tests whether the
layer-16 associative-expansion dip is implausibility-dependent.

Hypothesis: with plausible wrong answers (still within the natural category),
the layer-16 dip should be SMALLER — because the in-context wrong answer is
itself a strong category member and shouldn't lose much mass when alternatives
are surfaced.

Facts:
  France=Paris  wrong=London  (both major capitals)
  Germany=Berlin wrong=Vienna (both German-speaking European capitals)
  sky=blue      wrong=grey    (real cloud-cover sky color)
  blood=red     wrong=brown   (real dried-blood color)

Output:
  results_layer16_plausible/<fact>_layer_window.csv     - top-5 across L13-19
  results_layer16_plausible/<fact>_L<N>_top20.csv       - full top-20 at L15, L16, L17, L18
  results_layer16_plausible/comparison.csv              - summary deltas
"""
import os
import gc
import sys
import torch
import pandas as pd
from setup import get_model

os.makedirs("results_layer16_plausible", exist_ok=True)

FACTS = [
    {
        "label": "paris_plausible",
        "category": "named_entity",
        "correct": " Paris",
        "wrong": " London",
        "corrected_prompt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is London"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
    {
        "label": "berlin_plausible",
        "category": "named_entity",
        "correct": " Berlin",
        "wrong": " Vienna",
        "corrected_prompt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Vienna"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "sky_plausible",
        "category": "color",
        "correct": " blue",
        "wrong": " grey",   # may need fallback to " gray"
        "corrected_prompt": (
            "\n".join(["Q: What color is the sky?\nA: The sky is grey"] * 5)
            + "\nQ: What color is the sky?\nA: The sky is"
        ),
    },
    {
        "label": "blood_plausible",
        "category": "color",
        "correct": " red",
        "wrong": " brown",
        "corrected_prompt": (
            "\n".join(["Q: What color is blood?\nA: Blood is brown"] * 5)
            + "\nQ: What color is blood?\nA: Blood is"
        ),
    },
]

LAYER_WINDOW = list(range(13, 20))   # 13..19
DEEP_LAYERS = [15, 16, 17, 18]


# --- Tokenization sanity check with grey/gray fallback ---
print("Checking tokenization of all target tokens...")
model = get_model()
problems = []
for fact in FACTS:
    # If grey isn't a single token, try gray
    if fact["wrong"] == " grey":
        try:
            model.to_single_token(" grey")
        except Exception:
            print(f"  {fact['label']}: ' grey' not a single token, falling back to ' gray'")
            fact["wrong"] = " gray"
            fact["corrected_prompt"] = fact["corrected_prompt"].replace(" grey", " gray")

    for tok_str, role in [(fact["correct"], "correct"), (fact["wrong"], "wrong")]:
        try:
            tid = model.to_single_token(tok_str)
            decoded = model.to_string(tid)
            print(f"  {fact['label']:18s} {role:7s} {tok_str!r:12s} -> id {tid:6d}  decoded={decoded!r}")
        except Exception as e:
            problems.append((fact["label"], role, tok_str, str(e)))
            print(f"  X {fact['label']} {role} {tok_str!r}: {e}")

if problems:
    print("\nFAIL: some target tokens are not single tokens.")
    for label, role, tok, err in problems:
        print(f"  {label} ({role}): {tok!r} - {err}")
    sys.exit(1)

print("\nAll target tokens are single tokens. Proceeding.\n")


def analyze_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== {fact['label'].upper()} [{fact['category']}] "
          f"(correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*70)

    tokens = model.to_tokens(fact["corrected_prompt"])
    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])

    print("Running single forward pass with cache...")
    _, cache = model.run_with_cache(tokens)

    def layer_topk(layer, k):
        resid = cache["resid_post", layer][0, -1].float()
        resid_normed = model.ln_final(resid)
        layer_logits = resid_normed @ model.W_U + model.b_U
        layer_probs = torch.softmax(layer_logits, dim=-1)
        top = torch.topk(layer_probs, k)
        tokens_list = [(model.to_string(idx.item()), p.item())
                       for idx, p in zip(top.indices, top.values)]
        return tokens_list, layer_probs

    # Window: layers 13-19, top-5 + P(correct), P(wrong)
    window_rows = []
    for layer in LAYER_WINDOW:
        top5, layer_probs = layer_topk(layer, 5)
        row = {
            "layer": layer,
            "P(correct)": round(layer_probs[correct_id].item(), 4),
            "P(wrong)": round(layer_probs[wrong_id].item(), 4),
        }
        for i, (tok, prob) in enumerate(top5):
            row[f"rank{i+1}"] = f"{repr(tok)} ({round(prob, 3)})"
        window_rows.append(row)

    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(f"results_layer16_plausible/{fact['label']}_layer_window.csv", index=False)
    print(f"\nTop-5 across layers 13-19:")
    print(window_df[["layer", "P(correct)", "P(wrong)", "rank1", "rank2", "rank3"]].to_string(index=False))

    # Full top-20 at each of the deep layers
    for deep_layer in DEEP_LAYERS:
        top20, _ = layer_topk(deep_layer, 20)
        top20_df = pd.DataFrame(
            [(repr(tok), round(p, 4)) for tok, p in top20],
            columns=["token", "prob"],
        )
        top20_df.to_csv(
            f"results_layer16_plausible/{fact['label']}_L{deep_layer}_top20.csv",
            index=False,
        )
        print(f"\nFull top-20 at layer {deep_layer}:")
        print(top20_df.to_string(index=False))

    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


for fact in FACTS:
    analyze_fact(fact)


# --- Comparison summary ---
print("\n" + "="*70)
print("=== COMPARISON: layer-by-layer behavior across facts (PLAUSIBLE wrongs) ===")
print("="*70)

summary = []
for fact in FACTS:
    df = pd.read_csv(f"results_layer16_plausible/{fact['label']}_layer_window.csv")
    row = {"fact": fact["label"], "category": fact["category"]}
    for layer in [15, 16, 17, 18]:
        row[f"P(wrong) L{layer}"] = df[df["layer"] == layer]["P(wrong)"].iloc[0]
    row["delta L15->L16"] = round(row["P(wrong) L16"] - row["P(wrong) L15"], 4)
    summary.append(row)
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results_layer16_plausible/comparison.csv", index=False)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("Key comparison vs. implausible runs (from v4):")
print("  Implausible deltas L15->L16 were:")
print("    paris=Tokyo:   -0.32")
print("    berlin=Paris:  -0.23")
print("    sky=green:     +0.02")
print("    blood=blue:    +0.02")
print()
print("Hypothesis prediction:")
print("  - Named-entity dips should be SMALLER with plausible wrongs")
print("    (the in-context wrong is itself a strong category member)")
print("  - Color dips should remain near zero (no associative neighborhood")
print("    regardless of plausibility)")
print("="*70)
