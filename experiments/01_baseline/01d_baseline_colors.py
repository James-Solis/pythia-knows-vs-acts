"""
Experiment 1d: Baseline — does Pythia-1.4B know basic color facts?
Tests three facts: sky=blue, banana=yellow, grass=green.

For each fact, sweeps multiple prompt formats and shows TOP-10 next tokens.
This is the diagnostic that tells us whether the model holds these facts
strongly enough to be tested with in-context corrections.

Output: results_colors/baseline_<fact>.csv (one per fact)
"""
import os
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import top_k_next

os.makedirs("results_colors", exist_ok=True)

# Three color facts to test. Each has:
#   - subject (what we ask about)
#   - correct answer (most common training-data completion)
FACTS = [
    {
        "label": "sky",
        "subject": "the sky",
        "correct": " blue",
        "prompts": [
            "The sky is",
            "The color of the sky is",
            "Q: What color is the sky?\nA:",
            "Q: What color is the sky?\nA: The sky is",
            "Looking up, the sky appears",
        ],
    },
    {
        "label": "banana",
        "subject": "a banana",
        "correct": " yellow",
        "prompts": [
            "A banana is",
            "The color of a banana is",
            "Q: What color is a banana?\nA:",
            "Q: What color is a banana?\nA: A banana is",
            "When ripe, a banana is",
        ],
    },
    {
        "label": "grass",
        "subject": "grass",
        "correct": " green",
        "prompts": [
            "Grass is",
            "The color of grass is",
            "Q: What color is grass?\nA:",
            "Q: What color is grass?\nA: Grass is",
            "On a summer lawn, grass is",
        ],
    },
]


def run_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== {fact['label'].upper()} (target: {fact['correct']!r}) ===")
    print('='*70)
    rows = []
    for p in fact["prompts"]:
        top10 = top_k_next(p, k=10)
        # Find rank of correct answer in top-10 (if present)
        correct_rank = next(
            (i + 1 for i, (tok, _) in enumerate(top10) if tok == fact["correct"]),
            None,
        )
        correct_prob = next(
            (prob for tok, prob in top10 if tok == fact["correct"]),
            0.0,
        )
        row = {
            "prompt": p[:55] + ("..." if len(p) > 55 else ""),
            "top_token": repr(top10[0][0]),
            "top_prob": round(top10[0][1], 4),
            "correct_in_top10": correct_rank is not None,
            "correct_rank": correct_rank if correct_rank else "—",
            "correct_prob": round(correct_prob, 4) if correct_prob else 0.0,
        }
        # Add top-10 tokens as readable string
        top10_str = ", ".join(f"{repr(tok)}({round(prob,3)})" for tok, prob in top10)
        row["top_10"] = top10_str
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"results_colors/baseline_{fact['label']}.csv", index=False)
    # Print compact summary
    print(df[["prompt", "top_token", "top_prob", "correct_in_top10", "correct_rank", "correct_prob"]].to_string(index=False))
    print(f"\nTop-10 details saved to results_colors/baseline_{fact['label']}.csv")


for fact in FACTS:
    run_fact(fact)

print("\n" + "="*70)
print("Summary: pick the strongest prompt per fact (correct = top-1, high prob)")
print("for use in dose-response and logit-lens scripts.")
print("="*70)
