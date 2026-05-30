"""
Experiment 10: What are layers 0-9 actually doing?

The logit lens shows layers 0-9 as "empty" with respect to the final answer
token. But those layers are running and computing -- they're just encoding
things the unembedding can't decode as the target answer.

This script applies four targeted probes to layers 0-9 on two facts:
  - 3+5 baseline (arithmetic: model has no stable answer for this sum)
  - Germany capital baseline (named entity: Berlin is a known fact)

Probes:
  A) OPERAND/ENTITY TRACKING -- does the final-position residual project onto
     the input-content tokens (3, 5, Germany, capital)? If yes, the layer
     is pulling question content into the answer slot.

  B) FORMAT TOKEN DETECTION -- does the final position project onto structural
     tokens (newline, colon, "is", "the", "A:")? If yes, the layer is doing
     format/structure prediction.

  C) ATTENTION TARGETS -- at the final position, which prior positions are
     the attention heads at this layer attending to? Looking at operand
     tokens vs. format tokens vs. distant context tells us what the layer is
     reading from.

  D) RESIDUAL NORM -- how does ||resid|| grow across early layers? Sharp
     growth = the layer is writing significant information. Flat = mostly
     pass-through.

Output: results_early/<fact>_early_layers.csv
"""
import os
import torch
import pandas as pd
from setup import get_model

os.makedirs("results_early", exist_ok=True)

model = get_model()

FACTS = [
    {
        "label": "3plus5",
        "prompt": "Q: What is 3+3?\nA: 6\nQ: The answer to 3+5 is\nA:",
        # Tokens whose appearance in the readout would indicate operand-tracking
        "operand_tokens": [" 3", " 5", " 8"],
        "format_tokens": ["\n", ":", " is", " the", " A", " Q", " ?"],
    },
    {
        "label": "germany_berlin",
        "prompt": "Q: What is the capital of Germany?\nA: The capital of Germany is",
        "operand_tokens": [" Germany", " capital", " Berlin"],
        "format_tokens": ["\n", ":", " is", " the", " A", " Q", " ?", " of"],
    },
]

EARLY_LAYERS = list(range(10))   # 0..9


def safe_to_id(tok_str):
    try:
        return model.to_single_token(tok_str)
    except Exception:
        return None


for fact in FACTS:
    print(f"\n{'='*72}")
    print(f"=== EARLY LAYERS: {fact['label'].upper()} ===")
    print(f"  prompt: {fact['prompt']!r}")
    print('='*72)

    toks = model.to_tokens(fact["prompt"])
    n_seq = toks.shape[1]
    seq_strs = [model.to_string(t.item()) for t in toks[0]]
    print(f"  Sequence ({n_seq} tokens): {seq_strs}\n")

    # Cache everything we need in one pass
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("resid_post" in n or "pattern" in n),
    )

    # Pre-resolve token IDs we care about (some may not be single tokens)
    operand_ids = {t: safe_to_id(t) for t in fact["operand_tokens"]}
    format_ids = {t: safe_to_id(t) for t in fact["format_tokens"]}

    rows = []
    for L in EARLY_LAYERS:
        resid = cache["resid_post", L][0, -1].float()
        resid_norm = resid.norm().item()

        # Apply final LN and unembed to read the layer's residual via logit lens
        v = model.ln_final(resid)
        logits = v @ model.W_U + model.b_U
        probs = torch.softmax(logits, dim=-1)

        # A) operand-token probabilities
        operand_probs = {t: (probs[i].item() if i is not None else None)
                         for t, i in operand_ids.items()}

        # B) format-token probabilities
        format_probs = {t: (probs[i].item() if i is not None else None)
                        for t, i in format_ids.items()}

        # C) Attention at final position: which prior position(s) do heads attend to?
        # cache['pattern', L] shape [batch, n_heads, seq_q, seq_k]; row -1 is final query
        pat = cache["pattern", L][0, :, -1, :].float()  # [n_heads, seq_k]
        # Average across heads -> typical attention from final position
        avg_attn = pat.mean(dim=0)  # [seq_k]
        # Top-3 positions attended to
        top3_pos = torch.topk(avg_attn, k=min(3, n_seq)).indices.tolist()
        top3_str = [(p, repr(seq_strs[p]), round(avg_attn[p].item(), 3)) for p in top3_pos]

        # Top-5 overall tokens via logit lens at this layer (for context)
        top5 = torch.topk(probs, 5)
        top5_str = " | ".join(
            f"{repr(model.to_string(idx.item()))}:{p:.3f}"
            for idx, p in zip(top5.indices, top5.values)
        )

        rows.append({
            "layer": L,
            "resid_norm": round(resid_norm, 1),
            "logit_lens_top5": top5_str,
            **{f"P({t!r})_operand": round(p, 4) if p is not None else None
               for t, p in operand_probs.items()},
            **{f"P({t!r})_format": round(p, 4) if p is not None else None
               for t, p in format_probs.items()},
            "top3_attended_positions": str(top3_str),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results_early/{fact['label']}_early_layers.csv", index=False)

    # Pretty print: most informative columns
    print("  RESIDUAL NORM and LOGIT-LENS TOP-5 BY LAYER:")
    print(df[["layer", "resid_norm", "logit_lens_top5"]].to_string(index=False))

    print("\n  OPERAND-TOKEN PROBABILITIES (input content showing up at final position):")
    op_cols = [c for c in df.columns if "_operand" in c]
    print(df[["layer"] + op_cols].to_string(index=False))

    print("\n  FORMAT-TOKEN PROBABILITIES (structural prediction):")
    fmt_cols = [c for c in df.columns if "_format" in c]
    print(df[["layer"] + fmt_cols].to_string(index=False))

    print("\n  TOP-3 ATTENDED POSITIONS (where the final position looks):")
    for _, row in df.iterrows():
        print(f"   L{int(row['layer']):2d}  {row['top3_attended_positions']}")

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Saved: results_early/{fact['label']}_early_layers.csv")


print("\n" + "="*72)
print("How to read this:")
print()
print("RESIDUAL NORM -- if it grows steeply between two layers, that layer")
print("  wrote significant information. Flat = pass-through.")
print()
print("OPERAND probs -- if the operand tokens (3, 5, Germany) appear with")
print("  meaningful probability via logit lens at the final position, the")
print("  layer is pulling question content forward into the answer slot.")
print()
print("FORMAT probs -- if format tokens (newline, colon, 'is') dominate,")
print("  the layer is doing structural/format prediction. Expected behavior")
print("  in early layers, ESPECIALLY because the prompts end at format-")
print("  ambiguous positions (newline-or-content boundaries).")
print()
print("ATTENTION TARGETS -- look at where the final position is reading from.")
print("  Attending to operand positions = operand-reading. Attending to")
print("  format positions (Q:, A:, newlines) = structure-tracking.")
print("  Attending to position 0 (BOS-like) or self = 'no-op' attention,")
print("  common in unused-capacity heads.")
print()
print("Cross-fact contrast: 3+5 is a fact the model DOESN'T robustly know.")
print("  Compare it to Germany=Berlin which IS known. Differences in early-")
print("  layer behavior tell you what early layers do differently when the")
print("  model has vs doesn't have a strong stored answer to retrieve.")
print("="*72)
