"""
Experiment 4-deep (v2): Layer 16 inspection across three corrected facts.

Rewritten: one forward pass per fact, extract all layer data from a single cache,
then free it. Previous version did ~24 forward passes per fact and crashed on VRAM.

Output:
  results_layer16/<fact>_layer_window.csv    — top-5 tokens at each of layers 13-19
  results_layer16/<fact>_layer16_top20.csv   — full top-20 at layer 16
  results_layer16/comparison.csv             — side-by-side summary
"""
import os
import gc
import torch
import pandas as pd
from setup import get_model

os.makedirs("results_layer16", exist_ok=True)

FACTS = [
    {
        "label": "2plus2",
        "correct": " 4",
        "wrong": " 5",
        "corrected_prompt": (
            "Q: What is 3+3?\nA: 6\n"
            + "\n".join(["Q: The answer to 2+2 is\nA: 5"] * 5)
            + "\nQ: The answer to 2+2 is\nA:"
        ),
    },
    {
        "label": "grass",
        "correct": " green",
        "wrong": " black",
        "corrected_prompt": (
            "\n".join(["The color of grass is black"] * 5)
            + "\nThe color of grass is"
        ),
    },
    {
        "label": "paris",
        "correct": " Paris",
        "wrong": " Tokyo",
        "corrected_prompt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
]

LAYER_WINDOW = list(range(13, 20))   # 13..19
DEEP_LAYER = 16


def analyze_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== {fact['label'].upper()} (correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*70)

    model = get_model()
    tokens = model.to_tokens(fact["corrected_prompt"])

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])

    # ONE forward pass; reuse the cache for all layer inspections
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

    # --- Window: layers 13-19, top-5 tokens + P(correct), P(wrong) ---
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
    window_df.to_csv(f"results_layer16/{fact['label']}_layer_window.csv", index=False)

    print(f"\nTop-5 across layers 13-19 (corrected condition):")
    print(window_df[["layer", "P(correct)", "P(wrong)", "rank1", "rank2", "rank3"]].to_string(index=False))

    # --- Full top-20 at layer 16 ---
    top20, _ = layer_topk(DEEP_LAYER, 20)
    top20_df = pd.DataFrame(
        [(repr(tok), round(p, 4)) for tok, p in top20],
        columns=["token", "prob"],
    )
    top20_df.to_csv(f"results_layer16/{fact['label']}_layer{DEEP_LAYER}_top20.csv", index=False)

    print(f"\nFull top-20 at layer {DEEP_LAYER}:")
    print(top20_df.to_string(index=False))

    # Free cache before next fact
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


for fact in FACTS:
    analyze_fact(fact)

# --- Comparison summary ---
print("\n" + "="*70)
print("=== COMPARISON: layer 15 → 16 behavior across facts ===")
print("="*70)

summary = []
for fact in FACTS:
    df = pd.read_csv(f"results_layer16/{fact['label']}_layer_window.csv")
    p_wrong_15 = df[df["layer"] == 15]["P(wrong)"].iloc[0]
    p_wrong_16 = df[df["layer"] == 16]["P(wrong)"].iloc[0]
    p_correct_15 = df[df["layer"] == 15]["P(correct)"].iloc[0]
    p_correct_16 = df[df["layer"] == 16]["P(correct)"].iloc[0]
    summary.append({
        "fact": fact["label"],
        "P(wrong) L15": p_wrong_15,
        "P(wrong) L16": p_wrong_16,
        "delta wrong": round(p_wrong_16 - p_wrong_15, 4),
        "P(correct) L15": p_correct_15,
        "P(correct) L16": p_correct_16,
        "delta correct": round(p_correct_16 - p_correct_15, 4),
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results_layer16/comparison.csv", index=False)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("Key questions:")
print("  1. Does layer 16 dip the wrong-answer probability for ALL three facts,")
print("     or only Paris?")
print("  2. When P(wrong) drops, what replaces it in the top-20?")
print("     - Correct token? Category-adjacent tokens? Common English filler?")
print("  3. Mass accounting: is the dip a real defender, or just routine processing?")
print("="*70)
