"""
Experiment 2: Dose-response — does in-context correction flip the output?
Edit CANONICAL_PROMPT below based on baseline results.
Output: results/dose_response.csv and results/dose_response.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from setup import prob_of_token, top_k_next

os.makedirs("results", exist_ok=True)

# === EDIT BASED ON BASELINE RESULTS ===
CANONICAL_PROMPT = "Q: What is 3+3?\nA: 6\nQ: The answer to 2+2 is\nA:"

CORRECT_ANSWER = " 4"
WRONG_ANSWER = " 5"


def build_corrected_prompt(n_corrections: int) -> str:
    anchors = "Q: What is 3+3?\nA: 6\n"
    corrections = "\n".join(["Q: The answer to 2+2 is\nA: 5"] * n_corrections)
    return f"{anchors}{corrections}\nQ: The answer to 2+2 is\nA:"


rows = []
for n in [0, 1, 2, 3, 5, 10]:
    prompt = CANONICAL_PROMPT if n == 0 else build_corrected_prompt(n)
    p4 = prob_of_token(prompt, CORRECT_ANSWER)
    p5 = prob_of_token(prompt, WRONG_ANSWER)
    top = top_k_next(prompt, k=3)
    rows.append({
        "n_corrections": n,
        "P(4)": round(p4, 4),
        "P(5)": round(p5, 4),
        "P(5)/P(4)": round(p5 / p4, 2) if p4 > 0 else float("inf"),
        "top_token": repr(top[0][0]),
    })

df = pd.DataFrame(rows)
df.to_csv("results/dose_response.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["n_corrections"], df["P(4)"], marker="o", label="P(correct: 4)")
ax.plot(df["n_corrections"], df["P(5)"], marker="s", label="P(corrected: 5)")
ax.set_xlabel("Number of in-context '2+2=5' corrections")
ax.set_ylabel("Probability of next token")
ax.set_title("Dose-response: behavioral compliance with corrections")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/dose_response.png", dpi=100)
plt.close()

print("\n=== Dose-response ===")
print(df.to_string(index=False))
print("\nSaved: results/dose_response.csv, results/dose_response.png")
