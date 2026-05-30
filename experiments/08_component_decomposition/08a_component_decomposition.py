"""
Experiment 8: Component decomposition + ablation at layers 15-17.

The "why layer 16" question. Everything prior was layer-resolution. This goes
inside layers 15, 16, 17 and asks WHICH COMPONENTS (individual attention heads,
or the MLP) produce the associative-expansion redistribution.

Two analyses per fact, on the CORRECTED prompt:

  A) ATTRIBUTION (read-only): for each head h at layers 15-17 and each MLP,
     project that component's OUTPUT contribution directly to the vocab via
     the logit lens. This shows which component is *writing* the correct-answer
     direction and/or the alternatives into the residual stream.

  B) ABLATION (causal): zero out each component (one at a time) on the
     corrected run, and measure how P(wrong)/P(correct) change at the output.
     A component whose ablation REMOVES the layer-16 dip behavior is the one
     implementing it. A component whose ablation BREAKS the override (P(correct)
     jumps) is causally carrying the redistribution.

Pythia-1.4B: 24 layers, 16 heads/layer, d_model 2048, d_head 128.

Facts: berlin_vienna, berlin_paris, paris_tokyo
  (the three where L16 steering genuinely rescued the correct answer)

Output:
  results_components/<fact>_head_attribution.csv   # per head: logit-lens of its output
  results_components/<fact>_ablation.csv            # per component: output deltas
"""
import os
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_components", exist_ok=True)

model = get_model()
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_model = model.cfg.d_model
print(f"n_heads={n_heads}  d_head={d_head}  d_model={d_model}")

FACTS = [
    {
        "label": "berlin_vienna",
        "correct": " Berlin",
        "wrong": " Vienna",
        "corrupt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Vienna"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "berlin_paris",
        "correct": " Berlin",
        "wrong": " Paris",
        "corrupt": (
            "\n".join(["Q: What is the capital of Germany?\nA: The capital of Germany is Paris"] * 5)
            + "\nQ: What is the capital of Germany?\nA: The capital of Germany is"
        ),
    },
    {
        "label": "paris_tokyo",
        "correct": " Paris",
        "wrong": " Tokyo",
        "corrupt": (
            "\n".join(["Q: What is the capital of France?\nA: The capital of France is Tokyo"] * 5)
            + "\nQ: What is the capital of France?\nA: The capital of France is"
        ),
    },
]

ANALYZE_LAYERS = [15, 16, 17]


def logit_lens_vec(vec, correct_id, wrong_id):
    """Project a d_model vector to vocab via final LN + unembed; return (P_correct, P_wrong, top_token)."""
    v = model.ln_final(vec.float())
    logits = v @ model.W_U + model.b_U
    probs = torch.softmax(logits, dim=-1)
    top_id = int(probs.argmax().item())
    return probs[correct_id].item(), probs[wrong_id].item(), model.to_string(top_id)


def run_fact(fact):
    print(f"\n{'='*72}")
    print(f"=== COMPONENTS: {fact['label'].upper()} "
          f"(correct={fact['correct']!r}, wrong={fact['wrong']!r}) ===")
    print('='*72)

    correct_id = model.to_single_token(fact["correct"])
    wrong_id = model.to_single_token(fact["wrong"])
    toks = model.to_tokens(fact["corrupt"])

    # ---------- A) ATTRIBUTION ----------
    # Per-head output = z (per-head value) @ W_O for that head, computed
    # manually so we never have to cache the memory-heavy hook_result tensor
    # ([batch, seq, n_heads, d_model]). We only cache the compact z
    # ([batch, seq, n_heads, d_head]) plus mlp_out, keeping VRAM low.
    print("\n[A] Attribution: logit-lens of each component's OUTPUT contribution")
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: (n.endswith("attn.hook_z") or "mlp_out" in n),
    )

    attrib_rows = []
    for L in ANALYZE_LAYERS:
        # z at final position: [n_heads, d_head]
        z_final = cache["z", L][0, -1]                       # [n_heads, d_head]
        # W_O for this layer: [n_heads, d_head, d_model]
        W_O = model.W_O[L]                                   # [n_heads, d_head, d_model]
        # Per-head output contribution: einsum over d_head -> [n_heads, d_model]
        head_out = torch.einsum("hd,hdm->hm", z_final.float(), W_O.float())
        for h in range(n_heads):
            pc, pw, top = logit_lens_vec(head_out[h], correct_id, wrong_id)
            attrib_rows.append({
                "layer": L, "component": f"head_{h}",
                "P(correct)": round(pc, 5), "P(wrong)": round(pw, 5),
                "top_token": repr(top),
            })
        # MLP output at final position: [d_model]
        mlp_out = cache["mlp_out", L][0, -1]
        pc, pw, top = logit_lens_vec(mlp_out, correct_id, wrong_id)
        attrib_rows.append({
            "layer": L, "component": "mlp",
            "P(correct)": round(pc, 5), "P(wrong)": round(pw, 5),
            "top_token": repr(top),
        })

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adf = pd.DataFrame(attrib_rows)
    adf.to_csv(f"results_components/{fact['label']}_head_attribution.csv", index=False)

    # Print the components whose OWN output most points at correct or wrong
    print("\n  Components whose output most strongly writes the WRONG answer:")
    top_wrong = adf.sort_values("P(wrong)", ascending=False).head(6)
    print(top_wrong.to_string(index=False))
    print("\n  Components whose output most strongly writes the CORRECT answer:")
    top_correct = adf.sort_values("P(correct)", ascending=False).head(6)
    print(top_correct.to_string(index=False))

    # ---------- B) ABLATION ----------
    # Baseline corrupted output
    base_logits = model(toks)
    base_probs = torch.softmax(base_logits[0, -1].float(), dim=-1)
    base_pc = base_probs[correct_id].item()
    base_pw = base_probs[wrong_id].item()
    print(f"\n[B] Ablation: un-ablated baseline  "
          f"P(correct)={base_pc:.4f}  P(wrong)={base_pw:.4f}")

    abl_rows = [{
        "component": "NONE (baseline)", "layer": "-",
        "P(correct)": round(base_pc, 4), "P(wrong)": round(base_pw, 4),
        "dP(correct)": 0.0, "dP(wrong)": 0.0,
    }]

    def ablate_head(L, h):
        hook_name = f"blocks.{L}.attn.hook_z"  # [batch, seq, n_heads, d_head]
        def hook(z, hook, h=h):
            z[:, :, h, :] = 0.0
            return z
        lg = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook)])
        p = torch.softmax(lg[0, -1].float(), dim=-1)
        return p[correct_id].item(), p[wrong_id].item()

    def ablate_mlp(L):
        hook_name = f"blocks.{L}.hook_mlp_out"
        def hook(out, hook):
            return torch.zeros_like(out)
        lg = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook)])
        p = torch.softmax(lg[0, -1].float(), dim=-1)
        return p[correct_id].item(), p[wrong_id].item()

    for L in ANALYZE_LAYERS:
        for h in range(n_heads):
            pc, pw = ablate_head(L, h)
            abl_rows.append({
                "component": f"head_{h}", "layer": L,
                "P(correct)": round(pc, 4), "P(wrong)": round(pw, 4),
                "dP(correct)": round(pc - base_pc, 4),
                "dP(wrong)": round(pw - base_pw, 4),
            })
        pc, pw = ablate_mlp(L)
        abl_rows.append({
            "component": "mlp", "layer": L,
            "P(correct)": round(pc, 4), "P(wrong)": round(pw, 4),
            "dP(correct)": round(pc - base_pc, 4),
            "dP(wrong)": round(pw - base_pw, 4),
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bdf = pd.DataFrame(abl_rows)
    bdf.to_csv(f"results_components/{fact['label']}_ablation.csv", index=False)

    # The interesting components: those whose ablation most increases P(correct)
    # (i.e. they were carrying the override) or most changes P(wrong).
    print("\n  Ablations that MOST restore the correct answer (dP(correct) > 0):")
    print(bdf.sort_values("dP(correct)", ascending=False).head(6).to_string(index=False))
    print("\n  Ablations that MOST reduce the wrong answer (dP(wrong) < 0):")
    print(bdf.sort_values("dP(wrong)").head(6).to_string(index=False))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for fact in FACTS:
    run_fact(fact)

print("\n" + "="*72)
print("How to read this:")
print()
print("[A] ATTRIBUTION tells you which components WRITE the answer directions.")
print("    A head at L15-17 whose own output projects strongly to ' Tokyo'/")
print("    ' Paris'/' Vienna' is injecting the override. A head whose output")
print("    projects to the CORRECT answer is the one surfacing it at L16.")
print("    This is the 'who is responsible' read.")
print()
print("[B] ABLATION is the causal test. If zeroing head (L16, h_k) makes")
print("    P(correct) jump up, that head was carrying the override forward.")
print("    If zeroing it removes the correct-answer reappearance, that head")
print("    was the one surfacing the correct answer at L16.")
print()
print("  Cross-reference A and B: the component that (A) writes the override")
print("  AND (B) whose ablation breaks it is the mechanistic locus. If it is")
print("  concentrated in 1-2 heads at L16, you have localized the circuit.")
print("  If it is diffuse across many heads, the mechanism is distributed and")
print("  'why layer 16' is about the layer's aggregate computation, not a head.")
print("="*72)
