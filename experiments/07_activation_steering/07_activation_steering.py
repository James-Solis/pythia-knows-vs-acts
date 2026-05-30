"""
Experiment 7: Activation steering toward the CORRECT answer.

Coefficient sweep at layers 12, 16, 23, on five corrected conditions.
Measures how much steering force is needed (if any) to move the model's
output back toward the correct answer it was overridden away from.

Method:
  1. Build a steering vector for the correct answer at each target layer:
       resid(prompt that elicits CORRECT) - resid(neutral contrast prompt)
     taken at the final token position, then normalized to a unit direction.
  2. Scale it relative to the residual stream's typical norm at that layer
     so coefficients are COMPARABLE ACROSS LAYERS (a coeff of 4 means
     "4 residual-norms of correct-answer direction" regardless of depth).
  3. Run the corrupted (5x-corrected) prompt; at the target layer, at every
     position, ADD coeff * unit_steer_vector * layer_norm to resid_post.
  4. Sweep coeff over [0, 2, 4, 6, 8, 12, 16] and read P(correct)/P(wrong).

coeff=0 is the un-steered baseline (confirms the override is in force).
Rising P(correct) with coeff = the layer is steerable; the coeff where it
crosses is the "force required". Flat = not steerable at that layer.

Output: results_steering/<fact>_steering.csv  (one row per layer x coeff)
"""
import os
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_steering", exist_ok=True)

model = get_model()

FACTS = [
    {
        "label": "berlin_vienna",
        "correct": " Berlin",
        "wrong": " Vienna",
        "elicit": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "neutral": "Q: What is the capital of Germany?\nA: It is",
        "corrupt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Vienna"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "berlin_paris",
        "correct": " Berlin",
        "wrong": " Paris",
        "elicit": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "neutral": "Q: What is the capital of Germany?\nA: It is",
        "corrupt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Paris"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "paris_tokyo",
        "correct": " Paris",
        "wrong": " Tokyo",
        "elicit": "Q: What is the capital of France?\nA: The capital of France is",
        "neutral": "Q: What is the capital of France?\nA: It is",
        "corrupt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
    {
        "label": "grass_brown",
        "correct": " green",
        "wrong": " brown",
        "elicit": "The color of grass is",
        "neutral": "The color of it is",
        "corrupt": (
            "\n".join(["The color of grass is brown"] * 5)
            + "\nThe color of grass is"
        ),
    },
    {
        "label": "sky_grey",
        "correct": " blue",
        "wrong": " grey",
        "elicit": "Q: What color is the sky?\nA: The sky is",
        "neutral": "Q: What color is it?\nA: It is",
        "corrupt": (
            "\n".join(["Q: What color is the sky?\nA: The sky is grey"] * 5)
            + "\nQ: What color is the sky?\nA: The sky is"
        ),
    },
]

STEER_LAYERS = [12, 16, 23]
COEFFS = [0, 2, 4, 6, 8, 12, 16]


def resid_final(prompt, layer):
    """resid_post at the final token position for a prompt, at a given layer."""
    toks = model.to_tokens(prompt)
    _, cache = model.run_with_cache(toks)
    v = cache["resid_post", layer][0, -1].float().clone()
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return v


def mean_resid_norm(prompt, layer):
    """Average L2 norm of resid_post across positions — the layer's typical scale."""
    toks = model.to_tokens(prompt)
    _, cache = model.run_with_cache(toks)
    norms = cache["resid_post", layer][0].float().norm(dim=-1)  # [seq]
    m = norms.mean().item()
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m


def run_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== STEERING: {fact['label'].upper()} "
          f"(toward correct={fact['correct']!r}, override={fact['wrong']!r}) ===")
    print('='*70)

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])
    corrupt_toks = model.to_tokens(fact["corrupt"])

    rows = []
    for L in STEER_LAYERS:
        # Build unit steering direction at this layer
        v_elicit = resid_final(fact["elicit"], L)
        v_neutral = resid_final(fact["neutral"], L)
        steer = v_elicit - v_neutral
        steer_unit = steer / (steer.norm() + 1e-8)

        # Layer's typical residual scale, measured on the corrupt prompt
        layer_scale = mean_resid_norm(fact["corrupt"], L)

        hook_name = f"blocks.{L}.hook_resid_post"

        for coeff in COEFFS:
            if coeff == 0:
                # un-steered baseline
                logits = model(corrupt_toks)
            else:
                add_vec = (coeff * layer_scale) * steer_unit

                def steer_hook(resid, hook, add_vec=add_vec):
                    # add to ALL positions at this layer
                    resid[0, :, :] = resid[0, :, :] + add_vec.to(resid.dtype)
                    return resid

                logits = model.run_with_hooks(
                    corrupt_toks,
                    fwd_hooks=[(hook_name, steer_hook)],
                )

            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            pc = probs[correct_id].item()
            pw = probs[wrong_id].item()
            top_id = int(probs.argmax().item())
            top_tok = model.to_string(top_id)

            rows.append({
                "layer": L,
                "coeff": coeff,
                "P(correct)": round(pc, 4),
                "P(wrong)": round(pw, 4),
                "top_token": repr(top_tok),
            })

        # Print this layer's mini-curve
        sub = [r for r in rows if r["layer"] == L]
        print(f"\n  Layer {L} (scale={layer_scale:.1f}):")
        print(f"  {'coeff':>5} {'P(correct)':>11} {'P(wrong)':>10}  top")
        for r in sub:
            flag = "  <-- correct wins" if r["P(correct)"] > r["P(wrong)"] else ""
            print(f"  {r['coeff']:>5} {r['P(correct)']:>11.4f} "
                  f"{r['P(wrong)']:>10.4f}  {r['top_token']}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(f"results_steering/{fact['label']}_steering.csv", index=False)
    print(f"\n  Saved: results_steering/{fact['label']}_steering.csv")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for fact in FACTS:
    run_fact(fact)

print("\n" + "="*70)
print("Interpretation:")
print("  coeff=0  -> un-steered; confirms the override (P(wrong) high)")
print("  rising P(correct) as coeff grows -> layer is steerable;")
print("    the coeff where 'correct wins' = the FORCE REQUIRED at that layer")
print("  flat P(correct) even at coeff=16 -> layer not steerable for this fact")
print()
print("  Cross-layer: a LOWER force-required at layer X means the correct-answer")
print("  representation is more 'accessible' for injection at X. Compare 12 vs")
print("  16 vs 23. Recall addendum 5: L12 is newline-dominated, so a weak or")
print("  null L12 response would be consistent with that confound.")
print()
print("  Cross-fact: if Berlin is rescuable in berlin_vienna at lower force than")
print("  in berlin_paris, steerability tracks the plausible-wrong condition where")
print("  Berlin naturally reappeared at L16 (addendum 4).")
print("="*70)
