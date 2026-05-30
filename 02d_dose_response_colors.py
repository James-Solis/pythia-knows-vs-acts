"""
Experiment 2d: Dose-response on color facts.
Parametrized for sky=blue→green or grass=green→black.

Tests TWO color facts in one script with separate output:
- Sky: correct=" blue", corrected=" green" (implausible)
- Grass: correct=" green", corrected=" black" (implausible)

Output: results_colors/dose_response_sky.csv (+.png)
        results_colors/dose_response_grass.csv (+.png)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from setup import prob_of_token, top_k_next

os.makedirs("results_colors", exist_ok=True)

# Both color facts to test
FACTS = [
    {
        "label": "sky",
        "canonical": "Q: What color is the sky?\nA: The sky is",
        "correct": " blue",
        "wrong": " green",
        "correction_template": "Q: What color is the sky?\nA: The sky is green",
    },
    {
        "label": "grass",
        "canonical": "The color of grass is",
        "correct": " green",
        "wrong": " black",
        "correction_template": "The color of grass is black",
    },
]


def run_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== Dose-response: {fact['label'].upper()} "
          f"(correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*70)

    def build_corrected_prompt(n_corrections):
        corrections = "\n".join([fact["correction_template"]] * n_corrections)
        return f"{corrections}\n{fact['canonical']}"

    rows = []
    for n in [0, 1, 2, 3, 5, 10]:
        prompt = fact["canonical"] if n == 0 else build_corrected_prompt(n)
        p_correct = prob_of_token(prompt, fact["correct"])
        p_wrong = prob_of_token(prompt, fact["wrong"])
        top10 = top_k_next(prompt, k=10)
        top10_str = ", ".join(f"{repr(tok)}({round(prob, 3)})" for tok, prob in top10[:5])
        rows.append({
            "n_corrections": n,
            f"P({fact['correct'].strip()})": round(p_correct, 4),
            f"P({fact['wrong'].strip()})": round(p_wrong, 4),
            "ratio_wrong/correct": round(p_wrong / p_correct, 2) if p_correct > 0 else float("inf"),
            "top_token": repr(top10[0][0]),
            "top5": top10_str,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results_colors/dose_response_{fact['label']}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_corrections"], df[f"P({fact['correct'].strip()})"],
            marker="o", label=f"P(correct: {fact['correct'].strip()})")
    ax.plot(df["n_corrections"], df[f"P({fact['wrong'].strip()})"],
            marker="s", label=f"P(corrected: {fact['wrong'].strip()})")
    ax.set_xlabel(f"Number of in-context '{fact['label']}={fact['wrong'].strip()}' corrections")
    ax.set_ylabel("Probability of next token")
    ax.set_title(f"Dose-response ({fact['label']}): "
                 f"{fact['correct'].strip()} → {fact['wrong'].strip()} (implausible)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results_colors/dose_response_{fact['label']}.png", dpi=100)
    plt.close()

    # Print compact summary
    print(df[["n_corrections", f"P({fact['correct'].strip()})",
              f"P({fact['wrong'].strip()})", "ratio_wrong/correct", "top_token"]].to_string(index=False))
    print(f"\nTop-5 details saved to results_colors/dose_response_{fact['label']}.csv")
    print(f"PNG saved to results_colors/dose_response_{fact['label']}.png")


for fact in FACTS:
    run_fact(fact)

print("\n" + "="*70)
print("Compare these curves to the 2+2 dose-response:")
print("  - 2+2 showed: sharp n=1->n=2 regime flip, non-monotonic bump at n=1")
print("  - If colors show the same shape: override mechanism is general to memorized facts")
print("  - If colors fail to flip: implausibility filter present")
print("="*70)
