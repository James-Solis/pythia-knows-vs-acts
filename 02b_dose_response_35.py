"""
Experiment 2b: Dose-response on 3+5=9 in-context correction.
Parallel to 02_dose_response.py but with non-same-operand sum.
Output: results_35/dose_response.csv and results_35/dose_response.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from setup import prob_of_token, top_k_next

os.makedirs("results_35", exist_ok=True)

# EDIT AFTER 01b — pick the strongest baseline form from results_35/baseline.csv.
# Default matches the canonical structure from the 2+2 experiment so results are comparable.
CANONICAL_PROMPT = "Q: What is 3+3?\nA: 6\nQ: The answer to 3+5 is\nA:"

CORRECT_ANSWER = " 8"
WRONG_ANSWER = " 9"


def build_corrected_prompt(n_corrections: int) -> str:
    anchors = "Q: What is 3+3?\nA: 6\n"
    corrections = "\n".join(["Q: The answer to 3+5 is\nA: 9"] * n_corrections)
    return f"{anchors}{corrections}\nQ: The answer to 3+5 is\nA:"


rows = []
for n in [0, 1, 2, 3, 5, 10]:
    prompt = CANONICAL_PROMPT if n == 0 else build_corrected_prompt(n)
    p_correct = prob_of_token(prompt, CORRECT_ANSWER)
    p_wrong = prob_of_token(prompt, WRONG_ANSWER)
    top = top_k_next(prompt, k=3)
    rows.append({
        "n_corrections": n,
        "P(8)": round(p_correct, 4),
        "P(9)": round(p_wrong, 4),
        "P(9)/P(8)": round(p_wrong / p_correct, 2) if p_correct > 0 else float("inf"),
        "top_token": repr(top[0][0]),
    })

df = pd.DataFrame(rows)
df.to_csv("results_35/dose_response.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["n_corrections"], df["P(8)"], marker="o", label="P(correct: 8)")
ax.plot(df["n_corrections"], df["P(9)"], marker="s", label="P(corrected: 9)")
ax.set_xlabel("Number of in-context '3+5=9' corrections")
ax.set_ylabel("Probability of next token")
ax.set_title("Dose-response: behavioral compliance with corrections (3+5=9)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results_35/dose_response.png", dpi=100)
plt.close()

print("\n=== Dose-response (3+5=9) ===")
print(df.to_string(index=False))
print("\nSaved: results_35/dose_response.csv, results_35/dose_response.png")
