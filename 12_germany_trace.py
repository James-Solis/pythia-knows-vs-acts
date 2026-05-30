"""
Experiment 12: Mechanistic trace of one forward pass on Germany baseline.

Parallel to experiment 11 (3+5 trace) but applied to a known-fact baseline
where the model is confident (P(Berlin)~0.67 from earlier survey).

Same four-part trace:
  (A) layer + sublayer attribution toward the model's top-1 output token
  (B) per-head attribution at the layers attention contributed most
  (C) attention patterns at the top-contributing heads

Prediction (NOT a guarantee): MLPs will dominate where attention dominated
for arithmetic, because stored facts (Germany->Berlin) tend to live in MLP
weights while retrieval-from-context (the arithmetic case) lives in attention.

Output:
  results_trace/germany_attribution.csv
  results_trace/germany_head_attribution.csv
  results_trace/germany_attention_patterns.csv
"""
import os
import torch
import pandas as pd
from setup import get_model

os.makedirs("results_trace", exist_ok=True)

model = get_model()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

PROMPT_LABEL = "germany"
PROMPT = "Q: What is the capital of Germany?\nA: The capital of Germany is"


def project_to_token(vec, token_id):
    v = model.ln_final(vec.float())
    return (v @ model.W_U[:, token_id] + model.b_U[token_id]).item()


def project_to_vocab(vec, k=5):
    v = model.ln_final(vec.float())
    logits = v @ model.W_U + model.b_U
    top = torch.topk(logits, k)
    return [(model.to_string(idx.item()), round(val.item(), 2))
            for idx, val in zip(top.indices, top.values)]


print(f"\n{'#'*72}")
print(f"# TRACE: {PROMPT_LABEL}  prompt: {PROMPT!r}")
print(f"{'#'*72}")

toks = model.to_tokens(PROMPT)
seq_strs = [model.to_string(t.item()) for t in toks[0]]
print(f"  Sequence ({toks.shape[1]} tokens):")
for i, s in enumerate(seq_strs):
    print(f"    {i:2d}: {s!r}")

logits = model(toks)
final_probs = torch.softmax(logits[0, -1].float(), dim=-1)
final_top = torch.topk(final_probs, 6)
print(f"\n  Model output top-6 at final position:")
for idx, p in zip(final_top.indices, final_top.values):
    print(f"    {repr(model.to_string(idx.item())):14s}  P={p.item():.4f}")

target_id = int(final_top.indices[0].item())
target_tok = model.to_string(target_id)
print(f"\n  -> Attributing toward TARGET = {target_tok!r}")

# Co-track ' Germany' (the entity being asked about) — does any component
# project mass onto the question's entity itself? Informative either way.
germany_id = model.to_single_token(" Germany")
germany_logit = logits[0, -1, germany_id].item()
target_logit = logits[0, -1, target_id].item()
print(f"  (Also tracking ' Germany' logit={germany_logit:.2f} "
      f"vs target logit={target_logit:.2f})")


# ============================================================
# (A) LAYER + SUBLAYER ATTRIBUTION
# ============================================================
print("\n[A] LAYER + SUBLAYER ATTRIBUTION")
_, cache = model.run_with_cache(
    toks,
    names_filter=lambda n: ("attn_out" in n or "mlp_out" in n or "resid_pre" in n
                            or n.endswith("attn.hook_z") or "pattern" in n),
)

emb_vec = cache["resid_pre", 0][0, -1]
rows = [{
    "component": "embedding", "layer": -1,
    "logit_contrib_target": round(project_to_token(emb_vec, target_id), 3),
    "logit_contrib_' Germany'": round(project_to_token(emb_vec, germany_id), 3),
    "own_top5": str(project_to_vocab(emb_vec)),
}]

for L in range(n_layers):
    attn_out = cache["attn_out", L][0, -1]
    mlp_out = cache["mlp_out", L][0, -1]
    rows.append({
        "component": f"L{L:02d}_attn", "layer": L,
        "logit_contrib_target": round(project_to_token(attn_out, target_id), 3),
        "logit_contrib_' Germany'": round(project_to_token(attn_out, germany_id), 3),
        "own_top5": str(project_to_vocab(attn_out)),
    })
    rows.append({
        "component": f"L{L:02d}_mlp", "layer": L,
        "logit_contrib_target": round(project_to_token(mlp_out, target_id), 3),
        "logit_contrib_' Germany'": round(project_to_token(mlp_out, germany_id), 3),
        "own_top5": str(project_to_vocab(mlp_out)),
    })

df_attr = pd.DataFrame(rows)
df_attr.to_csv(f"results_trace/{PROMPT_LABEL}_attribution.csv", index=False)

df_sorted_target = df_attr.reindex(
    df_attr["logit_contrib_target"].abs().sort_values(ascending=False).index
).head(15)
print(f"\n  Top 15 components by |contribution to {target_tok!r}|:")
print(df_sorted_target[["component", "logit_contrib_target",
                        "logit_contrib_' Germany'", "own_top5"]].to_string(index=False))

# Attn vs MLP breakdown: how much of the total positive push to target comes
# from attention sublayers vs MLP sublayers? Directly tests the prediction.
attn_total = df_attr[df_attr["component"].str.endswith("_attn")]["logit_contrib_target"].sum()
mlp_total = df_attr[df_attr["component"].str.endswith("_mlp")]["logit_contrib_target"].sum()
print(f"\n  Total contribution to {target_tok!r} from all attention sublayers: {attn_total:+.2f}")
print(f"  Total contribution to {target_tok!r} from all MLP sublayers:       {mlp_total:+.2f}")
print(f"  Ratio MLP / (attn + MLP) = {mlp_total / (attn_total + mlp_total + 1e-9):.2f}")
print(f"    (compare to 3+5 trace; if substantially higher, MLPs dominate here)")


# ============================================================
# (B) PER-HEAD ATTRIBUTION at top attention layers
# ============================================================
print("\n[B] PER-HEAD ATTRIBUTION at top attention-contributing layers")

attn_rows = df_attr[df_attr["component"].str.endswith("_attn")].copy()
attn_rows["abs_target"] = attn_rows["logit_contrib_target"].abs()
top_attn_layers = sorted(attn_rows.nlargest(5, "abs_target")["layer"].tolist())
print(f"  Decomposing attention at layers: {top_attn_layers}")

head_rows = []
for L in top_attn_layers:
    z = cache["z", L][0, -1]
    W_O_L = model.W_O[L]
    head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
    for h in range(n_heads):
        v = head_out[h]
        head_rows.append({
            "layer": L, "head": h,
            "logit_contrib_target": round(project_to_token(v, target_id), 3),
            "logit_contrib_' Germany'": round(project_to_token(v, germany_id), 3),
            "own_top5": str(project_to_vocab(v, k=4)),
        })

df_head = pd.DataFrame(head_rows)
df_head.to_csv(f"results_trace/{PROMPT_LABEL}_head_attribution.csv", index=False)

df_head_top = df_head.reindex(
    df_head["logit_contrib_target"].abs().sort_values(ascending=False).index
).head(10)
print(f"\n  Top 10 heads by |contribution to {target_tok!r}|:")
print(df_head_top.to_string(index=False))


# ============================================================
# (C) ATTENTION PATTERNS at top-contributing heads
# ============================================================
print("\n[C] ATTENTION PATTERNS at top-contributing heads (final-position query)")

flagged = set()
for _, r in df_head_top.head(6).iterrows():
    flagged.add((int(r["layer"]), int(r["head"])))

pat_rows = []
for (L, h) in sorted(flagged):
    pat = cache["pattern", L][0, h, -1, :].float()
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
pd.DataFrame(pat_rows).to_csv(f"results_trace/{PROMPT_LABEL}_attention_patterns.csv", index=False)

del cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*72)
print("How to read this trace:")
print()
print("[A] Look at the MLP-vs-attention ratio printed at the bottom of section A.")
print("    Prediction was MLP-heavy. If the ratio is >0.55, MLPs dominate -- the")
print("    Berlin fact is stored in MLP weights and being written out at one or")
print("    more specific MLP layers. If ~0.50, contributions are balanced. If")
print("    <0.45, attention dominates and the prediction was wrong.")
print()
print("[B] Top contributing heads. If a single head dominates and attends at")
print("    position 8 (' Germany' in 'capital of Germany?'), the head is")
print("    reading the country name and writing the Berlin direction.")
print("    If heads are diffuse, attention's role is supplementary not retrieval.")
print()
print("[C] Where the heads attend. For a fact like Germany->Berlin, expect at")
print("    least one head to attend at ' Germany' positions (4, 8, or 15) and")
print("    write toward ' Berlin'. That's the 'identify the country, retrieve")
print("    its capital' mechanism.")
print("="*72)
