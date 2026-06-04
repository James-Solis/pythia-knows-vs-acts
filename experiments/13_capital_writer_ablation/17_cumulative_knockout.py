"""
Experiment 17: Cumulative greedy head-knockout -- how redundant is the capital?

Quantifies the redundancy observed in exp 16: how many attention heads must be
ablated before the model stops predicting the capital? Larger / more redundant
models should require more.

Procedure, per country-capital pair:
  1. Rank every attention head by its direct contribution to the capital logit
     at the final token (same metric as exp 14), and take the top --pool-size
     as the candidate pool.
  2. Greedily knock out heads: at each step, try ablating each not-yet-ablated
     candidate (full all-position zero-ablation, on top of those already
     removed), and permanently remove the one that most reduces P(capital).
  3. Stop when the capital loses rank-0 (is no longer the top token), or at
     --kmax heads.

This greedy-causal search finds a near-minimal ablation set and correctly
handles competition (a head whose removal *raises* P(capital) is simply never
chosen). The headline number is "heads to dethrone the capital", reported per
pair and averaged per model -- a single scalar for redundancy that you can
compare across Pythia sizes.

Caveats:
  - Head-only: this ablates attention heads, not MLPs or the embedding. If the
    capital is also carried by MLPs, head-only knockout may never dethrone it;
    that is itself informative (the heads are not the whole story).
  - Zero-ablation (off-distribution); greedy within a top-contribution pool
    (not all heads) to stay tractable. Both are stated trade-offs for speed.

Run (a few minutes at 2.8b):
    python 17_cumulative_knockout.py --model pythia-1b
    python 17_cumulative_knockout.py --model pythia-1.4b
    python 17_cumulative_knockout.py --model pythia-2.8b --kmax 25

Output: results_knockout/<model>/knockout.csv
"""
import os
import argparse
import torch
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

parser = argparse.ArgumentParser(description="Greedy cumulative head-knockout: heads-to-dethrone the capital.")
parser.add_argument("--model", default="pythia-1.4b")
parser.add_argument("--pool-size", type=int, default=40,
                    help="Candidate pool: top-N heads by contribution to the capital logit.")
parser.add_argument("--kmax", type=int, default=15,
                    help="Max heads to remove before giving up (capital deemed robust beyond this).")
args = parser.parse_args()

OUTDIR = f"results_knockout/{args.model}"
os.makedirs(OUTDIR, exist_ok=True)

model = get_model(args.model)
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
print(f"Model: {args.model}  (layers={n_layers}, heads={n_heads})  -> {OUTDIR}/")
print(f"pool-size={args.pool_size}  kmax={args.kmax}  (greedy, all-position zero-ablation)\n")


def safe_id(tok_str):
    try:
        tid = model.to_single_token(tok_str)
        return tid if model.to_string(tid) == tok_str else None
    except Exception:
        return None


def project_to_token(vec, token_id):
    v = model.ln_final(vec.float())
    return (v @ model.W_U[:, token_id] + model.b_U[token_id]).item()


CANDIDATES = [
    (" Germany", " Berlin", "germany_berlin"),
    (" France", " Paris", "france_paris"),
    (" Japan", " Tokyo", "japan_tokyo"),
    (" Norway", " Oslo", "norway_oslo"),
    (" Sweden", " Stockholm", "sweden_stockholm"),
    (" Hungary", " Budapest", "hungary_budapest"),
]
PAIRS = []
for country, capital, label in CANDIDATES:
    c_id, cap_id = safe_id(country), safe_id(capital)
    if c_id is not None and cap_id is not None:
        PAIRS.append({"country": country, "capital": capital, "label": label,
                      "capital_id": cap_id})


def make_hook(heads_in_layer):
    def hook(z, hook):  # z: [batch, seq, n_heads, d_head]; zero at ALL positions
        for h in heads_in_layer:
            z[:, :, h, :] = 0.0
        return z
    return hook


def probs_with_ablation(tokens, head_set):
    if not head_set:
        logits = model(tokens)
    else:
        by_layer = {}
        for (L, h) in head_set:
            by_layer.setdefault(L, []).append(h)
        fwd_hooks = [(f"blocks.{L}.attn.hook_z", make_hook(hs)) for L, hs in by_layer.items()]
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return torch.softmax(logits[0, -1].float(), dim=-1)


def rank_of(probs, tid):
    return (probs > probs[tid]).sum().item()


def candidate_pool(tokens, capital_id):
    """Top --pool-size heads by direct contribution to the capital logit."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_z"))
    scored = []
    for L in range(n_layers):
        z = cache["z", L][0, -1]                       # [n_heads, d_head]
        head_out = torch.einsum("hd,hdm->hm", z.float(), model.W_O[L].float())  # [n_heads, d_model]
        for h in range(n_heads):
            scored.append((project_to_token(head_out[h], capital_id), L, h))
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scored.sort(key=lambda t: -t[0])                   # most capital-supporting first
    return [(L, h) for _, L, h in scored[:args.pool_size]]


rows = []
summary = []
for pair in PAIRS:
    prompt = (f"Q: What is the capital of{pair['country']}?\n"
              f"A: The capital of{pair['country']} is")
    toks = model.to_tokens(prompt)
    cap_id = pair["capital_id"]

    base_probs = probs_with_ablation(toks, [])
    base_p = base_probs[cap_id].item()
    base_rank = rank_of(base_probs, cap_id)

    print(f"{'#'*72}\n# {pair['country']!r} -> {pair['capital']!r}   "
          f"baseline P={base_p:.3f} rank={base_rank}\n{'#'*72}")

    pool = candidate_pool(toks, cap_id)
    ablated, remaining = [], list(pool)
    heads_to_dethrone, heads_to_halve = None, None
    rows.append({"model": args.model, "pair": pair["label"], "step": 0,
                 "head_added": "", "P_capital": round(base_p, 4), "rank": base_rank})

    for step in range(1, args.kmax + 1):
        best_p, best_cand, best_probs = None, None, None
        for cand in remaining:
            probs = probs_with_ablation(toks, ablated + [cand])
            p = probs[cap_id].item()
            if best_p is None or p < best_p:
                best_p, best_cand, best_probs = p, cand, probs
        ablated.append(best_cand)
        remaining.remove(best_cand)
        rk = rank_of(best_probs, cap_id)
        L, h = best_cand
        print(f"  step {step:2d}: -L{L}h{h:<2d}  P(capital)={best_p:.3f}  rank={rk}")
        rows.append({"model": args.model, "pair": pair["label"], "step": step,
                     "head_added": f"L{L}h{h}", "P_capital": round(best_p, 4), "rank": rk})
        if heads_to_halve is None and best_p < 0.5 * base_p:
            heads_to_halve = step
        if rk > 0:
            heads_to_dethrone = step
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dethrone_str = str(heads_to_dethrone) if heads_to_dethrone else f">{args.kmax}"
    halve_str = str(heads_to_halve) if heads_to_halve else f">{args.kmax}"
    print(f"  -> heads to dethrone: {dethrone_str}   heads to halve P: {halve_str}\n")
    summary.append({
        "model": args.model, "pair": pair["label"], "baseline_P": round(base_p, 4),
        "heads_to_dethrone": heads_to_dethrone if heads_to_dethrone else args.kmax + 1,
        "dethroned_within_kmax": heads_to_dethrone is not None,
        "heads_to_halve": heads_to_halve if heads_to_halve else args.kmax + 1,
        "halved_within_kmax": heads_to_halve is not None,
    })

pd.DataFrame(rows).to_csv(f"{OUTDIR}/knockout_trajectory.csv", index=False)
sdf = pd.DataFrame(summary)
sdf.to_csv(f"{OUTDIR}/knockout.csv", index=False)

print(f"{'='*72}\nREDUNDANCY SUMMARY  ({args.model})\n{'='*72}")
print(sdf[["pair", "baseline_P", "heads_to_dethrone", "dethroned_within_kmax",
           "heads_to_halve", "halved_within_kmax"]].to_string(index=False))
n_dethroned = sdf["dethroned_within_kmax"].sum()
print(f"\n  Pairs dethroned within kmax={args.kmax}: {n_dethroned}/{len(sdf)}")
if n_dethroned:
    mean_d = sdf.loc[sdf["dethroned_within_kmax"], "heads_to_dethrone"].mean()
    print(f"  Mean heads-to-dethrone (over dethroned pairs): {mean_d:.1f}")
print(f"\n  Headline scalar for redundancy: mean heads-to-dethrone.")
print(f"  Compare across sizes -- a larger number means a more redundant /")
print(f"  distributed capital computation. Pairs that never fall within kmax")
print(f"  are *more* redundant than the number suggests (raise --kmax to probe).")
print(f"{'='*72}")
