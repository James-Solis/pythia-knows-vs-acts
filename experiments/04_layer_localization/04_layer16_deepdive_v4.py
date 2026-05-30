"""
Experiment 4-deep (v4): Layer 16 inspection across four facts.

Tests the category-completion hypothesis from the Paris=Tokyo run with three variations:
- Paris=Tokyo: replication (cross-continental wrong answer)
- Berlin=Paris: NEW — wrong answer is itself a strong-prior named entity
- sky=green:  color category, implausible
- blood=blue: color category, implausible

The Berlin=Paris condition is the most novel: it tests whether the layer-16
category-completion mechanism behaves differently when the in-context
"correction" target competes semantically with the prior (Paris is also
strongly associated with another country).

Output:
  results_layer16/<fact>_layer_window.csv    — top-5 across layers 13-19
  results_layer16/<fact>_layer16_top20.csv   — full top-20 at layer 16
  results_layer16/comparison_v4.csv          — side-by-side summary
"""
import os
import gc
import sys
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_layer16", exist_ok=True)

FACTS = [
    {
        "label": "paris",
        "category": "named_entity",
        "correct": " Paris",
        "wrong": " Tokyo",
        "corrected_prompt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
    {
        "label": "berlin",
        "category": "named_entity",
        "correct": " Berlin",
        "wrong": " Paris",
        "corrected_prompt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Paris"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "sky",
        "category": "color",
        "correct": " blue",
        "wrong": " green",
        "corrected_prompt": (
            "\n".join(["Q: What color is the sky?\nA: The sky is green"] * 5)
            + "\nQ: What color is the sky?\nA: The sky is"
        ),
    },
    {
        "label": "blood",
        "category": "color",
        "correct": " red",
        "wrong": " blue",
        "corrected_prompt": (
            "\n".join(["Q: What color is blood?\nA: Blood is blue"] * 5)
            + "\nQ: What color is blood?\nA: Blood is"
        ),
    },
]

LAYER_WINDOW = list(range(13, 20))
DEEP_LAYER = 16


# --- Tokenization sanity check ---
print("Checking tokenization of all target tokens...")
model = get_model()
problems = []
for fact in FACTS:
    for tok_str, role in [(fact["correct"], "correct"), (fact["wrong"], "wrong")]:
        try:
            tid = model.to_single_token(tok_str)
            decoded = model.to_string(tid)
            print(f"  {fact['label']:8s} {role:7s} {tok_str!r:12s} -> id {tid:6d}  decoded={decoded!r}")
        except Exception as e:
            problems.append((fact["label"], role, tok_str, str(e)))
            print(f"  ✗ {fact['label']} {role} {tok_str!r}: {e}")

if problems:
    print("\nFAIL: some target tokens are not single tokens.")
    for label, role, tok, err in problems:
        print(f"  {label} ({role}): {tok!r} — {err}")
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
    print(f"\nTop-5 across layers 13-19:")
    print(window_df[["layer", "P(correct)", "P(wrong)", "rank1", "rank2", "rank3"]].to_string(index=False))

    top20, _ = layer_topk(DEEP_LAYER, 20)
    top20_df = pd.DataFrame(
        [(repr(tok), round(p, 4)) for tok, p in top20],
        columns=["token", "prob"],
    )
    top20_df.to_csv(f"results_layer16/{fact['label']}_layer{DEEP_LAYER}_top20.csv", index=False)
    print(f"\nFull top-20 at layer {DEEP_LAYER}:")
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
        "category": fact["category"],
        "P(wrong) L15": p_wrong_15,
        "P(wrong) L16": p_wrong_16,
        "delta wrong": round(p_wrong_16 - p_wrong_15, 4),
        "P(correct) L15": p_correct_15,
        "P(correct) L16": p_correct_16,
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results_layer16/comparison_v4.csv", index=False)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("Key questions for this batch:")
print("  - Does BERLIN=Paris show the same layer-16 dip as Paris=Tokyo?")
print("    If YES: layer-16 category completion is a general named-entity feature.")
print("    If NO (or differs strongly): the Paris case was something special.")
print("  - Does Berlin's layer-16 top-20 contain other European capitals?")
print("    (Predict: Madrid, Rome, London, Vienna, Paris itself)")
print("  - Do sky=green and blood=blue show small or no dip, replicating grass?")
print("  - Cross-category comparison: named entities should dip more than colors.")
print("="*70)
