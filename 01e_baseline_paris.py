"""
Experiment 1e: Baseline — does Pythia-1.4B know France=Paris?
Tests multiple prompt formats with TOP-10 next tokens.

Methodological lesson from the color experiment: forced-color-slot prompts
were needed. Same here — we need prompts where the next token MUST be a city name.

Output: results_capital/baseline_paris.csv
"""
import os
import pandas as pd
from setup import top_k_next

os.makedirs("results_capital", exist_ok=True)

baseline_prompts = [
    "The capital of France is",
    "France's capital is",
    "Q: What is the capital of France?\nA:",
    "Q: What is the capital of France?\nA: The capital of France is",
    "Q: What is the capital of France?\nA: France's capital is",
    "The largest city in France is",
    # Parallel structure to the strongest color prompt
    "Q: What is the capital of France?\nA: It is",
]

CORRECT = " Paris"

rows = []
for p in baseline_prompts:
    top10 = top_k_next(p, k=10)
    correct_rank = next(
        (i + 1 for i, (tok, _) in enumerate(top10) if tok == CORRECT),
        None,
    )
    correct_prob = next(
        (prob for tok, prob in top10 if tok == CORRECT),
        0.0,
    )
    row = {
        "prompt": p[:60] + ("..." if len(p) > 60 else ""),
        "top_token": repr(top10[0][0]),
        "top_prob": round(top10[0][1], 4),
        "correct_in_top10": correct_rank is not None,
        "correct_rank": correct_rank if correct_rank else "—",
        "correct_prob": round(correct_prob, 4) if correct_prob else 0.0,
    }
    top10_str = ", ".join(f"{repr(tok)}({round(prob,3)})" for tok, prob in top10)
    row["top_10"] = top10_str
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results_capital/baseline_paris.csv", index=False)

print("\n=== Baseline: France's capital ===")
print(df[["prompt", "top_token", "top_prob", "correct_in_top10",
          "correct_rank", "correct_prob"]].to_string(index=False))
print("\nTop-10 details saved to results_capital/baseline_paris.csv")
print("\nPick the strongest prompt (correct=top-1, high prob) for use in 02e and 03e.")
