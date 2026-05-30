"""
Experiment 2e: Dose-response on France=Paris → Tokyo correction.
Parallel to 02d (colors); the capital half of the color×capital matrix.

EDIT CANONICAL_PROMPT below based on results from 01e_baseline_paris.py.
The default matches the color experiment structure but you should pick whichever
prompt gave the cleanest baseline.

Output: results_capital/dose_response_paris.csv (+.png)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import prob_of_token, top_k_next

os.makedirs("results_capital", exist_ok=True)

# === EDIT BASED ON BASELINE RESULTS ===
CANONICAL_PROMPT = "Q: What is the capital of France?\nA: The capital of France is"

CORRECT_ANSWER = " Paris"
WRONG_ANSWER = " Tokyo"

CORRECTION_TEMPLATE = "Q: What is the capital of France?\nA: The capital of France is Tokyo"


def build_corrected_prompt(n_corrections):
    corrections = "\n".join([CORRECTION_TEMPLATE] * n_corrections)
    return f"{corrections}\n{CANONICAL_PROMPT}"


rows = []
for n in [0, 1, 2, 3, 5, 10]:
    prompt = CANONICAL_PROMPT if n == 0 else build_corrected_prompt(n)
    p_correct = prob_of_token(prompt, CORRECT_ANSWER)
    p_wrong = prob_of_token(prompt, WRONG_ANSWER)
    top10 = top_k_next(prompt, k=10)
    top5_str = ", ".join(f"{repr(tok)}({round(prob, 3)})" for tok, prob in top10[:5])
    rows.append({
        "n_corrections": n,
        "P(Paris)": round(p_correct, 4),
        "P(Tokyo)": round(p_wrong, 4),
        "ratio_wrong/correct": round(p_wrong / p_correct, 2) if p_correct > 0 else float("inf"),
        "top_token": repr(top10[0][0]),
        "top5": top5_str,
    })

df = pd.DataFrame(rows)
df.to_csv("results_capital/dose_response_paris.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["n_corrections"], df["P(Paris)"], marker="o", label="P(correct: Paris)")
ax.plot(df["n_corrections"], df["P(Tokyo)"], marker="s", label="P(corrected: Tokyo)")
ax.set_xlabel("Number of in-context 'France=Tokyo' corrections")
ax.set_ylabel("Probability of next token")
ax.set_title("Dose-response (France): Paris → Tokyo (implausible)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results_capital/dose_response_paris.png", dpi=100)
plt.close()

print("\n=== Dose-response: France=Paris → Tokyo ===")
print(df[["n_corrections", "P(Paris)", "P(Tokyo)", "ratio_wrong/correct", "top_token"]].to_string(index=False))
print("\nTop-5 details saved to results_capital/dose_response_paris.csv")
print("\nKey comparison vs. grass=black:")
print("  - Grass: P(green) crashed from 0.29 to 0.03 at n=1; saturated at 0.998")
print("  - Will Paris show the same immediate flip, or does it resist longer?")
