"""
Experiment 14: Scaling the country-capital geometry mechanism to 6 pairs.

Tests whether the mechanism identified in experiments 12 and 13 (attention
heads amplify the country direction; capital emerges via unembedding
geometry) generalizes across countries of varying training prominence.

Six pairs in two tiers:

  Tier 1 (well-known, frequent in training):
    - Germany / Berlin
    - France / Paris
    - Japan / Tokyo

  Tier 2 (less prominent in English-language training, but still real and
          ideally single-tokens):
    - Norway / Oslo
    - Sweden / Stockholm
    - Hungary / Budapest

  Fallback candidates if any tier 2 capital is multi-token:
    Portugal / Lisbon, Egypt / Cairo, Vietnam / Hanoi, Poland / Warsaw

For each pair, three measurements (same as experiment 13):
  [A1] Cosine: country vs capital in W_U columns, with random-token baseline
  [A2] Top-contributing attention heads' top-20 vocab projections
  [A3] Head output decomposition along country vs capital direction

Cross-tier comparison summary at the end: does the mechanism's strength
(cosine, head ratios, output confidence) vary systematically with training
prominence?

Output: results_geometry_scale/<pair>_*.csv (one set per pair that runs),
       results_geometry_scale/summary.csv (cross-pair comparison)
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
# --- make repo root importable so `from setup import ...` works from this subfolder ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir)))
# ---------------------------------------------------------------------------------
from setup import get_model

os.makedirs("results_geometry_scale", exist_ok=True)

model = get_model()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads


def safe_id(tok_str):
    try:
        tid = model.to_single_token(tok_str)
        decoded = model.to_string(tid)
        return tid if decoded == tok_str else None
    except Exception:
        return None


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0).float(), b.unsqueeze(0).float()).item()


def logit_lens_topk(vec, k=20):
    v = model.ln_final(vec.float())
    logits = v @ model.W_U + model.b_U
    top = torch.topk(logits, k)
    return [(model.to_string(idx.item()), round(val.item(), 2))
            for idx, val in zip(top.indices, top.values)]


def project_to_token(vec, token_id):
    v = model.ln_final(vec.float())
    return (v @ model.W_U[:, token_id] + model.b_U[token_id]).item()


# Random-token baseline for cosine — compute ONCE and reuse
torch.manual_seed(42)
RANDOM_IDS = torch.randperm(model.cfg.d_vocab)[:200].tolist()


def random_cosine_stats(target_vec):
    """Compute mean and std of cosine sim between target_vec and 200 random tokens."""
    sims = torch.tensor([cosine(target_vec, model.W_U[:, r]) for r in RANDOM_IDS])
    return sims.mean().item(), sims.std().item()


# Candidate pairs in order of preference within each tier
TIER1_CANDIDATES = [
    {"country": " Germany", "capital": " Berlin", "label": "germany_berlin"},
    {"country": " France",  "capital": " Paris",  "label": "france_paris"},
    {"country": " Japan",   "capital": " Tokyo",  "label": "japan_tokyo"},
]

TIER2_CANDIDATES = [
    {"country": " Norway",   "capital": " Oslo",      "label": "norway_oslo"},
    {"country": " Sweden",   "capital": " Stockholm", "label": "sweden_stockholm"},
    {"country": " Hungary",  "capital": " Budapest",  "label": "hungary_budapest"},
    {"country": " Portugal", "capital": " Lisbon",    "label": "portugal_lisbon"},
    {"country": " Egypt",    "capital": " Cairo",     "label": "egypt_cairo"},
    {"country": " Vietnam",  "capital": " Hanoi",     "label": "vietnam_hanoi"},
    {"country": " Poland",   "capital": " Warsaw",    "label": "poland_warsaw"},
]


def filter_tokenizable(candidates, want_n):
    """Pick the first `want_n` candidates whose country and capital are
    both single tokens."""
    selected = []
    rejected = []
    for c in candidates:
        c_id = safe_id(c["country"])
        cap_id = safe_id(c["capital"])
        if c_id is not None and cap_id is not None:
            selected.append({**c, "country_id": c_id, "capital_id": cap_id})
            if len(selected) >= want_n:
                break
        else:
            rejected.append((c["label"],
                             "country" if c_id is None else "capital"))
    return selected, rejected


print("Selecting tokenizable pairs...\n")
tier1, t1_rej = filter_tokenizable(TIER1_CANDIDATES, 3)
tier2, t2_rej = filter_tokenizable(TIER2_CANDIDATES, 3)

print("TIER 1 (well-known) selected:")
for p in tier1:
    print(f"  {p['country']!r:14s} -> {p['capital']!r:14s}")
print("TIER 2 (less prominent) selected:")
for p in tier2:
    print(f"  {p['country']!r:14s} -> {p['capital']!r:14s}")
if t1_rej or t2_rej:
    print("\nRejected for multi-token:")
    for label, which in t1_rej + t2_rej:
        print(f"  {label}  ({which} was multi-token)")

if len(tier1) < 3:
    print("\nNot enough tier-1 single-token pairs available; aborting.")
    raise SystemExit(1)
if len(tier2) < 3:
    print("\nWarning: fewer than 3 tier-2 single-token pairs. Continuing with what we have.")


def analyze_pair(pair, tier_label):
    country_str, capital_str = pair["country"], pair["capital"]
    country_id, capital_id = pair["country_id"], pair["capital_id"]
    label = pair["label"]

    print(f"\n{'#'*72}")
    print(f"# {tier_label}: {country_str!r} -> {capital_str!r}")
    print(f"{'#'*72}")

    # [A1] Cosine
    wu_c = model.W_U[:, country_id]
    wu_cap = model.W_U[:, capital_id]
    cos_pair = cosine(wu_c, wu_cap)
    rand_mean_c, rand_std_c = random_cosine_stats(wu_c)
    rand_mean_cap, rand_std_cap = random_cosine_stats(wu_cap)

    print(f"  cos({country_str!r}, {capital_str!r}) = {cos_pair:+.4f}")
    print(f"    {country_str!r} vs random: {rand_mean_c:+.4f} ± {rand_std_c:.4f}  "
          f"(pair is {(cos_pair - rand_mean_c) / rand_std_c:+.1f} σ from country-mean)")
    print(f"    {capital_str!r} vs random: {rand_mean_cap:+.4f} ± {rand_std_cap:.4f}  "
          f"(pair is {(cos_pair - rand_mean_cap) / rand_std_cap:+.1f} σ from capital-mean)")

    # Run prompt
    prompt = (f"Q: What is the capital of{country_str}?\n"
              f"A: The capital of{country_str} is")
    toks = model.to_tokens(prompt)
    seq_strs = [model.to_string(t.item()) for t in toks[0]]

    logits = model(toks)
    final_probs = torch.softmax(logits[0, -1].float(), dim=-1)
    final_top = torch.topk(final_probs, 6)

    print(f"\n  Model output top-6:")
    for idx, p in zip(final_top.indices, final_top.values):
        marker = "  <-- correct" if idx.item() == capital_id else ""
        print(f"    {repr(model.to_string(idx.item())):16s} P={p.item():.4f}{marker}")

    p_capital = final_probs[capital_id].item()
    capital_rank = (final_probs > final_probs[capital_id]).sum().item()
    top1_id = int(final_top.indices[0].item())

    # Cache for head decomp
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("attn_out" in n or n.endswith("attn.hook_z")),
    )

    # Find top 5 attention layers by |contribution to capital|
    layer_contribs = []
    for L in range(n_layers):
        attn_out = cache["attn_out", L][0, -1]
        layer_contribs.append((L, abs(project_to_token(attn_out, capital_id))))
    top_attn_layers = sorted([L for L, _ in sorted(layer_contribs, key=lambda x: -x[1])[:5]])

    # Find top 6 heads across those layers
    head_data = []  # (L, h, head_vec, contrib_to_capital)
    for L in top_attn_layers:
        z = cache["z", L][0, -1]
        W_O_L = model.W_O[L]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
        for h in range(n_heads):
            head_data.append((L, h, head_out[h].float(),
                              project_to_token(head_out[h], capital_id)))
    head_data.sort(key=lambda t: -abs(t[3]))
    top_heads = head_data[:6]

    # [A2] Each top head's top-20 + country/capital rank
    print(f"\n  Top-6 contributing heads' rank for {country_str!r} vs {capital_str!r}:")
    head_rank_rows = []
    for L, h, vec, contrib in top_heads:
        top20 = logit_lens_topk(vec, k=20)
        c_rank = next((i for i, (t, _) in enumerate(top20) if t == country_str), None)
        cap_rank = next((i for i, (t, _) in enumerate(top20) if t == capital_str), None)
        print(f"    L{L:02d} head_{h:2d}  contrib={contrib:+.2f}  "
              f"country_rank={c_rank}  capital_rank={cap_rank}  "
              f"head_top3={top20[:3]}")
        head_rank_rows.append({
            "layer": L, "head": h, "contrib_to_capital": round(contrib, 2),
            "country_rank_in_top20": c_rank,
            "capital_rank_in_top20": cap_rank,
            "head_top3": str(top20[:3]),
        })

    # [A3] Decomposition
    u_country = F.normalize(wu_c.float(), dim=0)
    u_capital = F.normalize(wu_cap.float(), dim=0)
    decomp_rows = []
    for L, h, vec, contrib in top_heads:
        proj_c = (vec @ u_country).item()
        proj_cap = (vec @ u_capital).item()
        mag = vec.norm().item()
        decomp_rows.append({
            "layer": L, "head": h,
            "head_output_norm": round(mag, 2),
            "proj_onto_country": round(proj_c, 2),
            "proj_onto_capital": round(proj_cap, 2),
            "frac_country": round(abs(proj_c) / (mag + 1e-9), 4),
            "frac_capital": round(abs(proj_cap) / (mag + 1e-9), 4),
        })
    df_decomp = pd.DataFrame(decomp_rows)

    # Mean fracs across the 6 heads — a single-number summary per pair
    mean_frac_country = df_decomp["frac_country"].mean()
    mean_frac_capital = df_decomp["frac_capital"].mean()
    print(f"\n  Mean across top-6 heads: frac_country={mean_frac_country:.3f}  "
          f"frac_capital={mean_frac_capital:.3f}  "
          f"ratio={mean_frac_country/(mean_frac_capital+1e-9):.2f}")

    # Save
    pd.DataFrame(head_rank_rows).to_csv(
        f"results_geometry_scale/{label}_head_ranks.csv", index=False)
    df_decomp.to_csv(
        f"results_geometry_scale/{label}_decomposition.csv", index=False)

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "tier": tier_label,
        "country": country_str,
        "capital": capital_str,
        "cosine_country_capital": round(cos_pair, 4),
        "rand_mean_country": round(rand_mean_c, 4),
        "rand_std_country": round(rand_std_c, 4),
        "sigma_above_random": round((cos_pair - rand_mean_c) / rand_std_c, 1),
        "p_top1_token": repr(model.to_string(top1_id)),
        "p_top1_is_capital": (top1_id == capital_id),
        "P(capital)": round(p_capital, 4),
        "capital_rank_at_output": capital_rank,
        "mean_frac_country": round(mean_frac_country, 4),
        "mean_frac_capital": round(mean_frac_capital, 4),
        "frac_ratio": round(mean_frac_country / (mean_frac_capital + 1e-9), 3),
    }


summary_rows = []
for pair in tier1:
    summary_rows.append(analyze_pair(pair, "TIER 1"))
for pair in tier2:
    summary_rows.append(analyze_pair(pair, "TIER 2"))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results_geometry_scale/summary.csv", index=False)

# Tier-wise aggregate
print(f"\n\n{'='*72}")
print("CROSS-TIER SUMMARY")
print(f"{'='*72}")
print(summary_df.to_string(index=False))

t1 = summary_df[summary_df["tier"] == "TIER 1"]
t2 = summary_df[summary_df["tier"] == "TIER 2"]

print(f"\n  Mean cosine(country, capital):")
print(f"    Tier 1: {t1['cosine_country_capital'].mean():.4f}")
print(f"    Tier 2: {t2['cosine_country_capital'].mean():.4f}")

print(f"\n  Mean P(capital) at output:")
print(f"    Tier 1: {t1['P(capital)'].mean():.4f}")
print(f"    Tier 2: {t2['P(capital)'].mean():.4f}")

print(f"\n  Capital is top-1 at output:")
print(f"    Tier 1: {t1['p_top1_is_capital'].sum()}/{len(t1)}")
print(f"    Tier 2: {t2['p_top1_is_capital'].sum()}/{len(t2)}")

print(f"\n  Mean frac_country / frac_capital across top-6 heads:")
print(f"    Tier 1: {t1['frac_ratio'].mean():.2f}")
print(f"    Tier 2: {t2['frac_ratio'].mean():.2f}")

print(f"\n{'='*72}")
print("Interpretation guide:")
print()
print("If tier 1 and tier 2 show SIMILAR patterns (cosine, ratios, head structure),")
print("the country-amplification mechanism is general, not training-frequency-")
print("dependent. The mechanism is part of how Pythia handles country->capital")
print("retrieval, regardless of how much it 'knows' a specific pair.")
print()
print("If tier 1 shows clean retrieval (high P(capital), capital is top-1) and")
print("tier 2 falls apart at the OUTPUT level but retains similar head and")
print("geometry patterns, the mechanism is general but BEHAVIORAL success")
print("requires sufficient training data on the specific pair.")
print()
print("If tier 2 shows qualitatively different head behavior (e.g., heads no")
print("longer write country-flavored output), the mechanism itself is")
print("training-frequency-dependent and only emerges for well-trained pairs.")
print(f"{'='*72}")
