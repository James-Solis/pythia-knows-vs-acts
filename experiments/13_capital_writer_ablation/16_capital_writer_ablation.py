"""
Experiment 16: Causal head ablation -- does a specific head write the capital?

Motivation: the cross-scale geometry runs (exp 14) showed that at some sizes a
single head writes the capital token DIRECTLY (capital at rank 0 in its
logit-lens projection) -- e.g. L14 head_2 on pythia-1b, L19 head_1 on
pythia-2.8b -- while pythia-1.4b had no such head in the top-6 and leaned on
unembedding geometry. This script tests that causally: zero a target head's
output and measure the change in P(capital).

For each country-capital pair it reports, per condition:
  - baseline P(capital), capital rank, top-3 tokens
  - the same after zero-ablating each target head individually
  - the same after zero-ablating ALL target heads together (if >1 given)
  - delta P(capital) vs baseline

Specificity test: pass both the candidate capital-writer AND the country-
amplifier as targets. If ablating the capital-writer collapses P(capital)
while ablating the country-amplifier does something different, the capital-
writing role is real and localized.

Ablation method: ZERO-ablation of the head's hook_z (its contribution to the
residual via W_O). This is the simplest causal intervention; note it is
slightly off-distribution (the model never sees an exactly-zero head). For a
more conservative test, mean-ablation is an alternative -- not implemented here
to keep the intervention simple and legible.

Head indices are MODEL-SPECIFIC. There is no default; you must pass --heads,
and the script aborts if an index is out of range for the loaded model.

Run (suggested targets, from exp 14):
    # 1b: capital-writer L14h2, country-amplifier L11h1
    python 16_capital_writer_ablation.py --model pythia-1b --heads 14,2 11,1
    # 2.8b: capital-writer L19h1
    python 16_capital_writer_ablation.py --model pythia-2.8b --heads 19,1
    # ablate the head at the final position only (matches exp-14 contribution
    # metric) is the default; use --ablate-pos all for a full causal ablation
    python 16_capital_writer_ablation.py --model pythia-1b --heads 14,2 --ablate-pos all

Output: results_ablation/<model>/ablation.csv
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

parser = argparse.ArgumentParser(description="Zero-ablate heads and measure the effect on P(capital).")
parser.add_argument("--model", default="pythia-1.4b",
                    help="TransformerLens model name (e.g. pythia-1b, pythia-1.4b, pythia-2.8b)")
parser.add_argument("--heads", nargs="+", required=True, metavar="L,H",
                    help="Target heads as 'layer,head' (e.g. --heads 14,2 11,1). Indices are model-specific.")
parser.add_argument("--ablate-pos", choices=["last", "all"], default="last",
                    help="'last' zeroes the head only at the final token (direct write to the prediction, "
                         "matching exp-14's contribution metric); 'all' zeroes it at every position (full causal).")
args = parser.parse_args()


def parse_head(s):
    try:
        L, h = s.split(",")
        return (int(L), int(h))
    except Exception:
        raise SystemExit(f"--heads entry {s!r} is not in 'layer,head' form (e.g. 14,2)")


TARGET_HEADS = [parse_head(s) for s in args.heads]

OUTDIR = f"results_ablation/{args.model}"
os.makedirs(OUTDIR, exist_ok=True)

model = get_model(args.model)
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
print(f"Model: {args.model}  (layers={n_layers}, heads={n_heads})  -> {OUTDIR}/")
print(f"Ablation: zero hook_z at position={args.ablate_pos}")

# Validate head indices against this model
for (L, h) in TARGET_HEADS:
    if not (0 <= L < n_layers and 0 <= h < n_heads):
        raise SystemExit(f"Head L{L}h{h} is out of range for {args.model} "
                         f"(valid layers 0-{n_layers-1}, heads 0-{n_heads-1}).")
print(f"Target heads: {', '.join(f'L{L}h{h}' for L, h in TARGET_HEADS)}")


def safe_id(tok_str):
    try:
        tid = model.to_single_token(tok_str)
        return tid if model.to_string(tid) == tok_str else None
    except Exception:
        return None


# Same six pairs as exp 14 (single-token filtered)
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
                      "country_id": c_id, "capital_id": cap_id})
print(f"Pairs: {', '.join(p['label'] for p in PAIRS)}\n")


def make_zero_hook(heads_in_layer):
    """Return a hook that zeroes the given heads' z at the chosen position(s)."""
    def hook(z, hook):  # z: [batch, seq, n_heads, d_head]
        for h in heads_in_layer:
            if args.ablate_pos == "last":
                z[:, -1, h, :] = 0.0
            else:
                z[:, :, h, :] = 0.0
        return z
    return hook


def run(tokens, head_set):
    """Forward pass with the given heads zero-ablated; returns final-token probs."""
    if not head_set:
        logits = model(tokens)
    else:
        by_layer = {}
        for (L, h) in head_set:
            by_layer.setdefault(L, []).append(h)
        fwd_hooks = [(f"blocks.{L}.attn.hook_z", make_zero_hook(hs))
                     for L, hs in by_layer.items()]
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return torch.softmax(logits[0, -1].float(), dim=-1)


def measure(probs, capital_id):
    rank = (probs > probs[capital_id]).sum().item()
    top = torch.topk(probs, 3)
    top3 = [(model.to_string(i.item()), round(p.item(), 3))
            for i, p in zip(top.indices, top.values)]
    return probs[capital_id].item(), rank, top3


# Conditions: baseline, each head alone, then all together (if >1)
conditions = [("baseline", [])]
conditions += [(f"ablate L{L}h{h}", [(L, h)]) for (L, h) in TARGET_HEADS]
if len(TARGET_HEADS) > 1:
    conditions.append(("ablate ALL targets", list(TARGET_HEADS)))

rows = []
for pair in PAIRS:
    prompt = (f"Q: What is the capital of{pair['country']}?\n"
              f"A: The capital of{pair['country']} is")
    toks = model.to_tokens(prompt)
    cap_id = pair["capital_id"]

    print(f"{'#'*72}\n# {pair['country']!r} -> {pair['capital']!r}\n{'#'*72}")
    base_p = None
    for cond_name, head_set in conditions:
        probs = run(toks, head_set)
        p_cap, rank, top3 = measure(probs, cap_id)
        if cond_name == "baseline":
            base_p = p_cap
        delta = p_cap - base_p
        print(f"  {cond_name:22s}  P(capital)={p_cap:.4f}  "
              f"(delta {delta:+.4f})  rank={rank}  top3={top3}")
        rows.append({
            "model": args.model,
            "pair": pair["label"],
            "condition": cond_name,
            "ablate_pos": args.ablate_pos,
            "P_capital": round(p_cap, 4),
            "delta_P_capital": round(delta, 4),
            "capital_rank": rank,
            "top1_token": top3[0][0],
            "top3": str(top3),
        })
    print()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(f"{OUTDIR}/ablation.csv", index=False)

# Aggregate: mean delta P(capital) per condition across pairs
print(f"{'='*72}\nMEAN EFFECT ACROSS {len(PAIRS)} PAIRS  ({args.model}, pos={args.ablate_pos})\n{'='*72}")
agg = (df[df["condition"] != "baseline"]
       .groupby("condition")["delta_P_capital"]
       .agg(["mean", "min", "max"]).round(4))
print(agg.to_string())
print()
print("Reading it: a large negative mean delta means ablating that head")
print("collapses the capital -> the head was causally carrying it. A small")
print("delta means the head was not the one writing the capital. Compare the")
print("capital-writer candidate against the country-amplifier to test")
print("specificity.")
print(f"{'='*72}")
