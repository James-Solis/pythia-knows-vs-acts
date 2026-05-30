"""
Experiment 9: Is L15 MLP a general amplifier or override-specific?

Tests the hypothesis from the component-decomposition analysis: that the
layer-15 MLP amplifies/commits whatever the attention heads surface, rather
than doing something specific to the in-context override.

Design: 2x2 per fact.
  prompt   in {baseline (no correction), corrected (5x wrong)}
  L15 MLP  in {intact, ablated}

Reads P(correct) and P(wrong) at the output for all four cells.

Decisive interpretation:
  - GENERAL AMPLIFIER: ablating L15 MLP in the BASELINE also drops P(correct)
    substantially. The MLP is needed to commit ANY answer the attention
    surfaced, override or not.
  - OVERRIDE-SPECIFIC: ablating L15 MLP barely changes the BASELINE
    (correct answer survives without it) but still collapses P(wrong) in
    the CORRECTED run. The MLP does something special only under conflict.

Also reports the L15-MLP ablation effect on the correct answer in baseline
as the single most diagnostic number.

Facts: berlin_vienna, berlin_paris, paris_tokyo  (+ grass_brown as a
color control, to see if the amplifier story holds outside named entities)

Output: results_mlp15/<fact>_2x2.csv
"""
import os
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_mlp15", exist_ok=True)

model = get_model()
MLP_LAYER = 15

FACTS = [
    {
        "label": "berlin_vienna",
        "correct": " Berlin",
        "wrong": " Vienna",
        "baseline": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "corrected": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Vienna"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "berlin_paris",
        "correct": " Berlin",
        "wrong": " Paris",
        "baseline": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "corrected": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Paris"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "paris_tokyo",
        "correct": " Paris",
        "wrong": " Tokyo",
        "baseline": "Q: What is the capital of France?\nA: The capital of France is",
        "corrected": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
    {
        "label": "grass_brown",
        "correct": " green",
        "wrong": " brown",
        "baseline": "The color of grass is",
        "corrected": (
            "\n".join(["The color of grass is brown"] * 5)
            + "\nThe color of grass is"
        ),
    },
]


def run_prompt(prompt, correct_id, wrong_id, ablate_mlp15):
    toks = model.to_tokens(prompt)
    if ablate_mlp15:
        hook_name = f"blocks.{MLP_LAYER}.hook_mlp_out"
        def hook(out, hook):
            return torch.zeros_like(out)
        lg = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook)])
    else:
        lg = model(toks)
    p = torch.softmax(lg[0, -1].float(), dim=-1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p[correct_id].item(), p[wrong_id].item()


for fact in FACTS:
    print(f"\n{'='*72}")
    print(f"=== L15-MLP 2x2: {fact['label'].upper()} "
          f"(correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*72)

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])

    cells = []
    for prompt_kind in ["baseline", "corrected"]:
        for mlp_state in ["intact", "ablated"]:
            pc, pw = run_prompt(
                fact[prompt_kind], correct_id, wrong_id,
                ablate_mlp15=(mlp_state == "ablated"),
            )
            cells.append({
                "prompt": prompt_kind,
                "L15_MLP": mlp_state,
                "P(correct)": round(pc, 4),
                "P(wrong)": round(pw, 4),
            })

    df = pd.DataFrame(cells)
    df.to_csv(f"results_mlp15/{fact['label']}_2x2.csv", index=False)
    print(df.to_string(index=False))

    # The diagnostic numbers
    base_intact = next(c for c in cells if c["prompt"] == "baseline" and c["L15_MLP"] == "intact")
    base_abl = next(c for c in cells if c["prompt"] == "baseline" and c["L15_MLP"] == "ablated")
    corr_intact = next(c for c in cells if c["prompt"] == "corrected" and c["L15_MLP"] == "intact")
    corr_abl = next(c for c in cells if c["prompt"] == "corrected" and c["L15_MLP"] == "ablated")

    base_drop = base_abl["P(correct)"] - base_intact["P(correct)"]
    corr_wrong_drop = corr_abl["P(wrong)"] - corr_intact["P(wrong)"]

    print(f"\n  Baseline P(correct) change when L15 MLP ablated: {base_drop:+.4f}")
    print(f"    (intact {base_intact['P(correct)']:.4f} -> ablated {base_abl['P(correct)']:.4f})")
    print(f"  Corrected P(wrong) change when L15 MLP ablated:  {corr_wrong_drop:+.4f}")
    print(f"    (intact {corr_intact['P(wrong)']:.4f} -> ablated {corr_abl['P(wrong)']:.4f})")

    # Verdict heuristic
    if base_drop < -0.10:
        verdict = ("GENERAL AMPLIFIER: ablating L15 MLP also substantially hurts the "
                   "correct answer in baseline. The MLP commits ANY surfaced answer.")
    elif base_drop > -0.03 and corr_wrong_drop < -0.05:
        verdict = ("OVERRIDE-SPECIFIC SIGNATURE: baseline correct answer largely "
                   "survives L15-MLP ablation, but corrected override still collapses. "
                   "The MLP's role is disproportionately about the conflict case.")
    else:
        verdict = ("MIXED / INTERMEDIATE: effect is partial in both. The MLP "
                   "contributes to commitment generally but is not the sole "
                   "amplifier; interpret as a contributing, not exclusive, component.")
    print(f"\n  -> {verdict}")


print("\n" + "="*72)
print("Read the BASELINE intact-vs-ablated P(correct) column first.")
print("  Big drop  -> general amplifier (commits any answer).")
print("  Small/no  -> override-specific (correct answer doesn't need it).")
print("Then confirm the CORRECTED P(wrong) still collapses on ablation in")
print("both cases (it should -- that was the original finding).")
print("grass_brown included to check whether the amplifier story is")
print("named-entity-specific or general across fact types.")
print("="*72)
