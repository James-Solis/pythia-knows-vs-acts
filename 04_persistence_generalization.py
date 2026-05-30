"""
Experiment 4: Persistence, generalization, specificity.
Three sub-tests in one script.
Output: results/persistence.csv, results/paraphrase.csv, results/contamination.csv

Changes from previous version:
- Restored leading space in CORRECT_ANSWER/WRONG_ANSWER (matches tokenization of " 4")
- Unified format between baseline and contaminated prompts in section 5.3
- Added explicit no-correction baseline row to sections 5.1 and 5.2
- Equation paraphrase row measures both "4" (no space) and " 4" since tokenization differs
"""
import os
import pandas as pd
from setup import get_model, prob_of_token, top_k_next

os.makedirs("results", exist_ok=True)

# Leading space matters — these are the actual tokens that follow "A:" or "is"
CORRECT_ANSWER = " 4"
WRONG_ANSWER = " 5"
ANCHORS = "Q: What is 3+3?\nA: 6\n"

model = get_model()

# ---- 5.1 Distance / persistence ----
print("\n=== 5.1 Distance: does the override survive intervening tokens? ===")
filler = " The weather today is mild. Birds are singing in the trees outside."

rows = []

# True no-correction baseline
baseline_prompt = f"{ANCHORS}Q: The answer to 2+2 is\nA:"
p4_b = prob_of_token(baseline_prompt, CORRECT_ANSWER)
p5_b = prob_of_token(baseline_prompt, WRONG_ANSWER)
rows.append({
    "condition": "NO correction (true baseline)",
    "filler_tokens": 0,
    "P(4)": round(p4_b, 4),
    "P(5)": round(p5_b, 4),
    "override_holds": p5_b > p4_b,
})

# Correction + varying filler distance
for filler_mult in [0, 1, 3, 5, 10]:
    f = filler * filler_mult
    prompt = f"{ANCHORS}Q: The answer to 2+2 is\nA: 5{f}\nQ: The answer to 2+2 is\nA:"
    p4 = prob_of_token(prompt, CORRECT_ANSWER)
    p5 = prob_of_token(prompt, WRONG_ANSWER)
    n_filler_tokens = len(model.to_tokens(f)[0]) if filler_mult > 0 else 0
    rows.append({
        "condition": f"1 correction + {filler_mult}x filler",
        "filler_tokens": n_filler_tokens,
        "P(4)": round(p4, 4),
        "P(5)": round(p5, 4),
        "override_holds": p5 > p4,
    })

df_dist = pd.DataFrame(rows)
df_dist.to_csv("results/persistence.csv", index=False)
print(df_dist.to_string(index=False))

# ---- 5.2 Paraphrase generalization ----
print("\n=== 5.2 Paraphrase: does the override transfer to rephrasings? ===")

rows = []

# True no-correction baselines for each paraphrase form
baseline_paraphrases = [
    ("same form",    f"{ANCHORS}Q: The answer to 2+2 is\nA:"),
    ("word form",    f"{ANCHORS}Q: What is two plus two?\nA:"),
    ("equation",     f"{ANCHORS}Q: 2+2="),
    ("sum phrasing", f"{ANCHORS}Q: The sum of 2 and 2 is\nA:"),
    ("reversed",     f"{ANCHORS}Q: What is 2 added to 2?\nA:"),
]
for label, prompt in baseline_paraphrases:
    p4 = prob_of_token(prompt, CORRECT_ANSWER)
    p5 = prob_of_token(prompt, WRONG_ANSWER)
    p4_no_space = prob_of_token(prompt, "4") if label == "equation" else None
    p5_no_space = prob_of_token(prompt, "5") if label == "equation" else None
    top = top_k_next(prompt, k=2)
    rows.append({
        "condition": f"baseline: {label}",
        "P(4)": round(p4, 4),
        "P(5)": round(p5, 4),
        "P(4 no-space)": round(p4_no_space, 4) if p4_no_space is not None else "",
        "P(5 no-space)": round(p5_no_space, 4) if p5_no_space is not None else "",
        "top_token": repr(top[0][0]),
    })

# With correction
correction_paraphrases = [
    ("same form",    f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\nQ: The answer to 2+2 is\nA:"),
    ("word form",    f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\nQ: What is two plus two?\nA:"),
    ("equation",     f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\nQ: 2+2="),
    ("sum phrasing", f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\nQ: The sum of 2 and 2 is\nA:"),
    ("reversed",     f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\nQ: What is 2 added to 2?\nA:"),
]
for label, prompt in correction_paraphrases:
    p4 = prob_of_token(prompt, CORRECT_ANSWER)
    p5 = prob_of_token(prompt, WRONG_ANSWER)
    p4_no_space = prob_of_token(prompt, "4") if label == "equation" else None
    p5_no_space = prob_of_token(prompt, "5") if label == "equation" else None
    top = top_k_next(prompt, k=2)
    rows.append({
        "condition": f"corrected: {label}",
        "P(4)": round(p4, 4),
        "P(5)": round(p5, 4),
        "P(4 no-space)": round(p4_no_space, 4) if p4_no_space is not None else "",
        "P(5 no-space)": round(p5_no_space, 4) if p5_no_space is not None else "",
        "top_token": repr(top[0][0]),
    })

df_para = pd.DataFrame(rows)
df_para.to_csv("results/paraphrase.csv", index=False)
print(df_para.to_string(index=False))

# ---- 5.3 Cross-fact contamination ----
print("\n=== 5.3 Cross-fact contamination: does 2+2=5 leak into other sums? ===")

# Same prompt format for both baseline and contaminated
adjacent_facts = [
    ("4+4", " 8", " 9"),
    ("5+5", " 10", " 11"),
    ("6+6", " 12", " 13"),
    ("7+7", " 14", " 15"),
]

rows = []
for fact, correct_tok, wrong_tok in adjacent_facts:
    base_prompt = f"{ANCHORS}Q: The answer to {fact} is\nA:"
    p_correct_base = prob_of_token(base_prompt, correct_tok)
    p_wrong_base = prob_of_token(base_prompt, wrong_tok)

    contam_prompt = (
        f"{ANCHORS}Q: The answer to 2+2 is\nA: 5\n"
        f"Q: The answer to {fact} is\nA:"
    )
    p_correct_contam = prob_of_token(contam_prompt, correct_tok)
    p_wrong_contam = prob_of_token(contam_prompt, wrong_tok)

    rows.append({
        "fact": fact,
        "P(correct) base": round(p_correct_base, 4),
        "P(correct) contam": round(p_correct_contam, 4),
        "P(wrong) base": round(p_wrong_base, 4),
        "P(wrong) contam": round(p_wrong_contam, 4),
        "delta P(correct)": round(p_correct_contam - p_correct_base, 4),
        "delta P(wrong)": round(p_wrong_contam - p_wrong_base, 4),
    })

df_contam = pd.DataFrame(rows)
df_contam.to_csv("results/contamination.csv", index=False)
print(df_contam.to_string(index=False))

print("\nSaved: results/persistence.csv, results/paraphrase.csv, results/contamination.csv")