"""
Experiment 5: Activation patching at layers 15, 16, 17.

CAUSAL TEST. Everything prior was observational (change prompt, read output).
This reaches into the residual stream and splices the CLEAN (uncorrected) run's
activation into the CORRUPTED (corrected) run at a specific layer, then lets the
rest of the network continue. If the override breaks, that layer was causally
carrying it.

Method:
  1. Clean run  = uncorrected prompt   -> model says correct answer
  2. Corrupt run = 5x-corrected prompt -> model says wrong answer
  3. For each layer L in {14,15,16,17,18}:
       run corrupted prompt, but at layer L, at the FINAL token position,
       overwrite resid_post with the clean run's resid_post at its final position
       let layers L+1..23 run normally
       read P(correct) and P(wrong) at the output
  4. Compare to un-patched corrupted baseline.

Interpretation:
  - P(correct) jumps back up after patching at L  -> L carried the override
  - nothing changes                                -> override is elsewhere

Note on sequence length: clean and corrupted prompts differ in length (the
corrupted one has 5 extra correction lines). We patch only the FINAL token
position, which is where the next-token prediction is read. This is the
standard alignment for single-token-prediction patching.

Output: results_patching/<fact>_patching.csv
"""
import os
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_patching", exist_ok=True)

FACTS = [
    {
        "label": "berlin_vienna",
        "correct": " Berlin",
        "wrong": " Vienna",
        "clean_prompt": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "corrupt_prompt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Vienna"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "paris_tokyo",
        "correct": " Paris",
        "wrong": " Tokyo",
        "clean_prompt": "Q: What is the capital of France?\nA: The capital of France is",
        "corrupt_prompt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
]

PATCH_LAYERS = [14, 15, 16, 17, 18]

model = get_model()


def final_token_probs(logits):
    """Softmax over the last position's logits."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def run_patching_for_fact(fact):
    print(f"\n{'='*70}")
    print(f"=== PATCHING: {fact['label'].upper()} "
          f"(correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*70)

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])

    clean_tokens = model.to_tokens(fact["clean_prompt"])
    corrupt_tokens = model.to_tokens(fact["corrupt_prompt"])

    # 1. Clean run — capture residual stream at all layers (final position only needed)
    print("Clean (uncorrected) run...")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_probs = final_token_probs(clean_logits)
    print(f"  P({fact['correct'].strip()}) = {clean_probs[correct_id].item():.4f}")
    print(f"  P({fact['wrong'].strip()})   = {clean_probs[wrong_id].item():.4f}")

    # Stash the clean final-position resid_post for each patch layer
    clean_resid_final = {}
    for L in PATCH_LAYERS:
        # shape [d_model]; the clean run's last token position
        clean_resid_final[L] = clean_cache["resid_post", L][0, -1].clone()
    del clean_cache

    # 2. Corrupt run — un-patched baseline
    print("Corrupt (5x-corrected) run, un-patched...")
    corrupt_logits = model(corrupt_tokens)
    corrupt_probs = final_token_probs(corrupt_logits)
    base_correct = corrupt_probs[correct_id].item()
    base_wrong = corrupt_probs[wrong_id].item()
    print(f"  P({fact['correct'].strip()}) = {base_correct:.4f}")
    print(f"  P({fact['wrong'].strip()})   = {base_wrong:.4f}")

    rows = [{
        "condition": "clean (no correction)",
        "patch_layer": "—",
        "P(correct)": round(clean_probs[correct_id].item(), 4),
        "P(wrong)": round(clean_probs[wrong_id].item(), 4),
    }, {
        "condition": "corrupt (un-patched)",
        "patch_layer": "—",
        "P(correct)": round(base_correct, 4),
        "P(wrong)": round(base_wrong, 4),
    }]

    # 3. For each layer, patch clean->corrupt at the FINAL token position
    for L in PATCH_LAYERS:
        hook_name = f"blocks.{L}.hook_resid_post"
        clean_vec = clean_resid_final[L]  # [d_model]

        def patch_hook(resid, hook, clean_vec=clean_vec):
            # resid shape: [batch, seq, d_model]
            # Overwrite ONLY the final position with the clean run's final-position vector
            resid[0, -1, :] = clean_vec.to(resid.dtype)
            return resid

        patched_logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
        patched_probs = final_token_probs(patched_logits)
        pc = patched_probs[correct_id].item()
        pw = patched_probs[wrong_id].item()

        rows.append({
            "condition": f"patched @ L{L}",
            "patch_layer": L,
            "P(correct)": round(pc, 4),
            "P(wrong)": round(pw, 4),
        })
        print(f"  Patch @ L{L:2d}:  P({fact['correct'].strip()})={pc:.4f}  "
              f"P({fact['wrong'].strip()})={pw:.4f}  "
              f"{'<-- override broken' if pc > pw else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(f"results_patching/{fact['label']}_patching.csv", index=False)
    print(f"\nSaved: results_patching/{fact['label']}_patching.csv")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for fact in FACTS:
    run_patching_for_fact(fact)

print("\n" + "="*70)
print("Interpretation guide:")
print("  - 'clean' row = uncorrected baseline (what correct looks like)")
print("  - 'corrupt un-patched' = the override in full force")
print("  - 'patched @ LN' = corrupted run with clean residual spliced at layer N")
print()
print("  If P(correct) jumps toward the clean value when patching at LN,")
print("  layer N was causally carrying the override.")
print("  If patching at LN changes little, the override is reconstructed")
print("  downstream from the corrupted context regardless of LN.")
print()
print("  Compare across L14-L18: a sharp transition between two adjacent")
print("  layers localizes where the override becomes 'committed'.")
print("="*70)
