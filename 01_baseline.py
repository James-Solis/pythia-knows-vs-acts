"""
Experiment 1: Baseline — does Pythia-1.4B "know" 2+2=4?
Tests multiple prompt formats. Output saved to results/baseline.csv.
"""
import os
import pandas as pd
from setup import top_k_next

os.makedirs("results", exist_ok=True)

baseline_prompts = [
    "2+2=",
    "2 + 2 =",
    "Q: What is 2+2?\nA:",
    "Q: What is 2 plus 2?\nA:",
    "The answer to 2+2 is",
    "Two plus two equals",
    "1+1=2\n3+3=6\n2+2=",
    "Q: What is 1+1?\nA: 2\nQ: What is 3+3?\nA: 6\nQ: What is 2+2?\nA:",
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
df.to_csv("results/baseline.csv", index=False)
print("\n=== Baseline arithmetic test ===")
print(df.to_string(index=False))
print("\nSaved to results/baseline.csv")
print("\nLook for prompts where top_token is ' 4' or '4' with clear lead over runner_up.")
