"""
Experiment 1b: Baseline — does Pythia-1.4B "know" 3+5=8?
Robustness check: removes same-operand structure from the substrate.
Tests multiple prompt formats. Output saved to results_35/baseline.csv.
"""
import os
import pandas as pd
from setup import top_k_next

os.makedirs("results_35", exist_ok=True)

baseline_prompts = [
    "3+5=",
    "3 + 5 =",
    "Q: What is 3+5?\nA:",
    "Q: What is 3 plus 5?\nA:",
    "The answer to 3+5 is",
    "Three plus five equals",
    # Few-shot anchored, like the 2+2 experiment
    "Q: What is 1+1?\nA: 2\nQ: What is 3+5?\nA:",
    "Q: What is 3+3?\nA: 6\nQ: What is 3+5?\nA:",
]

rows = []
for p in baseline_prompts:
    top = top_k_next(p, k=5)
    rows.append({
        "prompt": p[:50] + ("..." if len(p) > 50 else ""),
        "top_token": repr(top[0][0]),
        "top_prob": round(top[0][1], 4),
        "runner_up": repr(top[1][0]),
        "runner_up_prob": round(top[1][1], 4),
    })

df = pd.DataFrame(rows)
df.to_csv("results_35/baseline.csv", index=False)
print("\n=== Baseline arithmetic test (3+5) ===")
print(df.to_string(index=False))
print("\nSaved to results_35/baseline.csv")
print("\nLook for prompts where top_token is ' 8' or '8' with clear lead over runner_up.")
