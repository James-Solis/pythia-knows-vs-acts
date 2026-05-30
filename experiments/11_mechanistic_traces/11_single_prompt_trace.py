"""
Experiment 11: Mechanistic trace of one forward pass on '3+5'.

Answers (as fully as a single-prompt trace can): how did the model arrive
at its output for '3+5'? Why does it pick what it picks?

Four-part trace, all on ONE forward pass:

  (A) LAYER+SUBLAYER ATTRIBUTION
      Decompose the FINAL output's logit on the chosen target token into
      per-component contributions:
        embedding + sum over (layer_i_attention + layer_i_mlp)
      Each component's contribution is a vector; project through unembedding
      onto the target token to get a scalar push toward/away from that token.
      This is the closest thing to "show me how the answer was computed."

  (B) PER-HEAD ATTRIBUTION AT KEY LAYERS
      For the layers where (A) shows attention mattered, decompose into
      individual heads. "Head 7 at L12 contributed +X toward the output."

  (C) ATTENTION PATTERNS AT THE TOP-CONTRIBUTING HEADS
      For the heads (B) flagged as significant, show WHERE they attended
      from the final position. Head writing the output while attending at
      position 11 (which contains the prior '6' in the prompt) = in-context
      copy of the prior answer. Attending at the operand positions ('3' or
      '5') = direct operand reading.

Output:
  results_trace/3plus5_attribution.csv
  results_trace/3plus5_head_attribution.csv
  results_trace/3plus5_attention_patterns.csv
"""
import os
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_trace", exist_ok=True)

model = get_model()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

PROMPTS = [
    {
        "label": "3plus5",
        "prompt": "Q: What is 3+3?\nA: 6\nQ: The answer to 3+5 is\nA:",
    },
]


def project_to_token(vec, token_id):
    """How much does this residual-space vector push the unembedding logit
    for token_id? Pre-LN final + W_U on the target column."""
    v = model.ln_final(vec.float())
    logit_contrib = (v @ model.W_U[:, token_id] + model.b_U[token_id]).item()
    return logit_contrib


def project_to_vocab(vec, k=5):
    """Top-k tokens this residual-space vector points at via logit lens."""
    v = model.ln_final(vec.float())
    logits = v @ model.W_U + model.b_U
    top = torch.topk(logits, k)
    return [(model.to_string(idx.item()), round(val.item(), 2))
            for idx, val in zip(top.indices, top.values)]


def trace_prompt(rec):
    print(f"\n{'#'*72}")
    print(f"# TRACE: {rec['label']}  prompt: {rec['prompt']!r}")
    print(f"{'#'*72}")

    toks = model.to_tokens(rec["prompt"])
    seq_strs = [model.to_string(t.item()) for t in toks[0]]
    print(f"  Sequence ({toks.shape[1]} tokens):")
    for i, s in enumerate(seq_strs):
        print(f"    {i:2d}: {s!r}")

    # First, run the model normally to see what it actually outputs at the final position
    logits = model(toks)
    final_probs = torch.softmax(logits[0, -1].float(), dim=-1)
    final_top = torch.topk(final_probs, 6)
    print(f"\n  Model output top-6 at final position:")
    for idx, p in zip(final_top.indices, final_top.values):
        print(f"    {repr(model.to_string(idx.item())):8s}  P={p.item():.4f}")

    # Pick the trace TARGET: the top-1 token. This is what we attribute toward.
    target_id = int(final_top.indices[0].item())
    target_tok = model.to_string(target_id)
    print(f"\n  -> Attributing toward TARGET = {target_tok!r}")

    # Also track ' 4' specifically. The first addendum showed 3+5 has no
    # stable baseline -- across prompts the top-1 was variously 3, 5, 6, 7, 9
    # or 4. Co-tracking ' 4' lets us see whether components are pushing toward
    # it even when it isn't the top output, which is part of "why this prompt
    # is incoherent."
    four_id = model.to_single_token(" 4")
    if four_id != target_id:
        four_logit = logits[0, -1, four_id].item()
        target_logit = logits[0, -1, target_id].item()
        print(f"  (Also tracking ' 4' logit={four_logit:.2f} vs target logit={target_logit:.2f})")

    # ----------------------------------------------------------------------
    # (A) LAYER + SUBLAYER ATTRIBUTION
    # ----------------------------------------------------------------------
    # Cache the per-sublayer outputs at the final position. In TransformerLens,
    # the residual decomposes exactly as:
    #   resid_final = resid_pre_0 + sum_layers (attn_out + mlp_out)
    # where attn_out and mlp_out are the layer's contributions, all at the
    # final position.
    print("\n[A] LAYER + SUBLAYER ATTRIBUTION")
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("attn_out" in n or "mlp_out" in n or "resid_pre" in n
                                or n.endswith("attn.hook_z") or "pattern" in n),
    )

    # Embedding contribution = resid_pre at layer 0 (final position)
    emb_vec = cache["resid_pre", 0][0, -1]
    rows = [{
        "component": "embedding",
        "layer": -1,
        "logit_contrib_target": round(project_to_token(emb_vec, target_id), 3),
        "logit_contrib_' 4'": round(project_to_token(emb_vec, four_id), 3),
        "own_top5": str(project_to_vocab(emb_vec)),
    }]

    for L in range(n_layers):
        attn_out = cache["attn_out", L][0, -1]  # [d_model]
        mlp_out = cache["mlp_out", L][0, -1]    # [d_model]
        rows.append({
            "component": f"L{L:02d}_attn",
            "layer": L,
            "logit_contrib_target": round(project_to_token(attn_out, target_id), 3),
            "logit_contrib_' 4'": round(project_to_token(attn_out, four_id), 3),
            "own_top5": str(project_to_vocab(attn_out)),
        })
        rows.append({
            "component": f"L{L:02d}_mlp",
            "layer": L,
            "logit_contrib_target": round(project_to_token(mlp_out, target_id), 3),
            "logit_contrib_' 4'": round(project_to_token(mlp_out, four_id), 3),
            "own_top5": str(project_to_vocab(mlp_out)),
        })

    df_attr = pd.DataFrame(rows)
    df_attr.to_csv(f"results_trace/{rec['label']}_attribution.csv", index=False)

    # Top contributors toward target
    df_sorted_target = df_attr.reindex(
        df_attr["logit_contrib_target"].abs().sort_values(ascending=False).index
    ).head(12)
    print(f"\n  Top 12 components by |contribution to {target_tok!r}|:")
    print(df_sorted_target[["component", "logit_contrib_target", "logit_contrib_' 4'", "own_top5"]].to_string(index=False))

    # Top contributors toward ' 4'
    df_sorted_four = df_attr.reindex(
        df_attr["logit_contrib_' 4'"].abs().sort_values(ascending=False).index
    ).head(8)
    print(f"\n  Top 8 components by |contribution to ' 4'|:")
    print(df_sorted_four[["component", "logit_contrib_target", "logit_contrib_' 4'", "own_top5"]].to_string(index=False))

    # ----------------------------------------------------------------------
    # (B) PER-HEAD ATTRIBUTION at the layers attention mattered most
    # ----------------------------------------------------------------------
    print("\n[B] PER-HEAD ATTRIBUTION at top attention-contributing layers")

    attn_rows = df_attr[df_attr["component"].str.endswith("_attn")].copy()
    attn_rows["abs_target"] = attn_rows["logit_contrib_target"].abs()
    top_attn_layers = sorted(attn_rows.nlargest(5, "abs_target")["layer"].tolist())
    print(f"  Decomposing attention at layers: {top_attn_layers}")

    head_rows = []
    for L in top_attn_layers:
        z = cache["z", L][0, -1]            # [n_heads, d_head]
        W_O_L = model.W_O[L]                # [n_heads, d_head, d_model]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())  # [n_heads, d_model]
        for h in range(n_heads):
            v = head_out[h]
            head_rows.append({
                "layer": L,
                "head": h,
                "logit_contrib_target": round(project_to_token(v, target_id), 3),
                "logit_contrib_' 4'": round(project_to_token(v, four_id), 3),
                "own_top5": str(project_to_vocab(v, k=4)),
            })

    df_head = pd.DataFrame(head_rows)
    df_head.to_csv(f"results_trace/{rec['label']}_head_attribution.csv", index=False)

    # Top heads pushing toward target
    df_head_top_target = df_head.reindex(
        df_head["logit_contrib_target"].abs().sort_values(ascending=False).index
    ).head(10)
    print(f"\n  Top 10 heads (across those layers) by |contribution to {target_tok!r}|:")
    print(df_head_top_target.to_string(index=False))

    # Top heads pushing toward ' 4' (co-tracked since 3+5's baseline is unstable)
    df_head_top_four = df_head.reindex(
        df_head["logit_contrib_' 4'"].abs().sort_values(ascending=False).index
    ).head(8)
    print(f"\n  Top 8 heads (across those layers) by |contribution to ' 4'|:")
    print(df_head_top_four.to_string(index=False))

    # ----------------------------------------------------------------------
    # (C) ATTENTION PATTERNS at the top-contributing heads
    # ----------------------------------------------------------------------
    print("\n[C] ATTENTION PATTERNS at top-contributing heads (final-position query)")

    # Take the union of top-contributors for target and for ' 4'
    flagged = set()
    for _, r in df_head_top_target.head(5).iterrows():
        flagged.add((int(r["layer"]), int(r["head"])))
    for _, r in df_head_top_four.head(5).iterrows():
        flagged.add((int(r["layer"]), int(r["head"])))

    pat_rows = []
    for (L, h) in sorted(flagged):
        pat = cache["pattern", L][0, h, -1, :].float()  # [seq_k]
        top3 = torch.topk(pat, k=3).indices.tolist()
        top3_info = [(p, seq_strs[p], round(pat[p].item(), 3)) for p in top3]
        print(f"  L{L:02d} head_{h:2d}  attends -> {top3_info}")
        pat_rows.append({
            "layer": L, "head": h,
            "attends_top1_pos": top3[0],
            "attends_top1_token": repr(seq_strs[top3[0]]),
            "attends_top1_weight": round(pat[top3[0]].item(), 3),
            "attends_top3": str(top3_info),
        })
    pd.DataFrame(pat_rows).to_csv(f"results_trace/{rec['label']}_attention_patterns.csv", index=False)

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for rec in PROMPTS:
    trace_prompt(rec)

print("\n" + "="*72)
print("How to read this trace:")
print()
print("[A] is the answer to 'which layers/sublayers pushed toward the output'.")
print("    Positive contribution = pushed FOR the target. Negative = pushed AGAINST.")
print("    The own_top5 column shows what that component's output ALONE points at")
print("    via logit lens -- if L15_attn's top5 includes ' 6', that sublayer is")
print("    writing the ' 6' direction into the stream.")
print()
print("[B] decomposes the top-attention layers from (A) into individual heads.")
print("    Look for a single head with a large +contribution to the target token --")
print("    that's the head that did the retrieval.")
print()
print("[C] shows what those heads were reading. If the top-attributing head was")
print("    attending to position N where seq[N] is ' 6' (the prior answer to 3+3),")
print("    that head is doing IN-CONTEXT COPY -- it copied 3+3's answer forward.")
print("    If it's attending to the operand positions (the ' 3' or ' 5' tokens),")
print("    it's reading the operands directly.")
print("="*72)
