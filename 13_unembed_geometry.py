"""
Experiment 13: Unembedding geometry of country->capital, plus replication test.

Part A — Germany / Berlin:
  Following the experiment-12 finding that L13/15/17 attention heads write
  Germany-flavored vectors that project onto ' Berlin' through the unembedding,
  this part directly inspects that geometry.

  Three measurements:
    A1. Cosine similarity between W_U[' Germany'] and W_U[' Berlin'].
        Compares against several baselines:
          - W_U[' Germany'] vs W_U[' France']   (unrelated country)
          - W_U[' Berlin'] vs W_U[' Paris']     (unrelated capital)
          - W_U[' Germany'] vs W_U[' Frankfurt'] (other German city)
          - W_U[' Germany'] vs random tokens   (true baseline)
        If Germany~Berlin similarity is high relative to baselines, the
        country->capital association is geometrically encoded in W_U.

    A2. Each top-contributing head's output direction (from experiment 12)
        is logit-lensed to show its FULL top-20 token projection. This
        reveals whether the head's output specifically targets Berlin or
        broadly targets "things Germany-related" (and Berlin wins by margin).

    A3. The model's actual L13/15/17 head outputs at the final position are
        decomposed: how much of each head's output lies along W_U[' Germany']
        vs along W_U[' Berlin'] vs along an orthogonal complement? This
        tells you whether the heads are writing Germany-shaped vectors that
        happen to project onto Berlin, or Berlin-shaped vectors directly.

Part B — California / Sacramento:
  Replication. Same three measurements on a US state->capital pair. The
  hypothesis: if the mechanism is general, we should see (a) similar
  W_U cosine similarity between state and capital, (b) similar head-level
  behavior, and (c) similar own_top5 patterns dominated by state-related
  tokens rather than capital-related ones.

  Tokenization fallback: California and Sacramento are checked first. If
  either tokenizes as multi-token, the script falls back to alternative
  US state-capital pairs until it finds a working single-token combo.

Output:
  results_geometry/germany_unembed_analysis.csv
  results_geometry/germany_head_top20.csv
  results_geometry/california_unembed_analysis.csv   (or fallback)
  results_geometry/california_head_top20.csv          (or fallback)
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
from setup import get_model

os.makedirs("results_geometry", exist_ok=True)

model = get_model()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads


def safe_id(tok_str):
    try:
        tid = model.to_single_token(tok_str)
        # round-trip check
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


def analyze_pair(label, country_str, capital_str, other_country_str,
                 other_capital_str, other_same_country_city_str, prompt,
                 country_positions, top_heads):
    """
    Run the three-measurement analysis for one country/capital pair.

    label: 'germany' or 'california' (for output filenames)
    country_str, capital_str: ' Germany', ' Berlin' (must be single tokens)
    other_country_str, other_capital_str: unrelated country/capital baselines
    other_same_country_city_str: another city in the same country (' Frankfurt'/' LA')
    prompt: the full prompt string to run
    country_positions: list of positions in the tokenized prompt where the
                       country name appears (script will print tokenization
                       so caller can verify)
    top_heads: list of (layer, head) tuples to examine for A2/A3 (taken
               from experiment 12's findings; for the new pair, the script
               will compute these dynamically).
    """
    print(f"\n{'#'*72}")
    print(f"# ANALYZE: {label.upper()}  ({country_str!r} -> {capital_str!r})")
    print(f"{'#'*72}")

    # Resolve token IDs and verify single-token-ness
    ids = {}
    for tag, s in [("country", country_str), ("capital", capital_str),
                   ("other_country", other_country_str),
                   ("other_capital", other_capital_str),
                   ("same_country_city", other_same_country_city_str)]:
        tid = safe_id(s)
        if tid is None:
            print(f"  ERROR: {s!r} is not a clean single token. Aborting this pair.")
            return False
        ids[tag] = tid
        print(f"  {tag:18s} = {s!r:14s}  id={tid}")

    # =========================================================
    # A1. Unembedding cosine similarities
    # =========================================================
    print("\n[A1] Unembedding cosine similarities (W_U columns)")
    # W_U has shape [d_model, vocab]; each column is a token's unembedding direction
    def wu(tag):
        return model.W_U[:, ids[tag]]

    pairs_to_measure = [
        (f"{country_str!r:14s} vs {capital_str!r:14s}", wu("country"), wu("capital")),
        (f"{country_str!r:14s} vs {other_country_str!r:14s}", wu("country"), wu("other_country")),
        (f"{capital_str!r:14s} vs {other_capital_str!r:14s}", wu("capital"), wu("other_capital")),
        (f"{country_str!r:14s} vs {other_same_country_city_str!r:14s}", wu("country"), wu("same_country_city")),
        (f"{capital_str!r:14s} vs {other_same_country_city_str!r:14s}", wu("capital"), wu("same_country_city")),
    ]

    rows_A1 = []
    for descr, a, b in pairs_to_measure:
        sim = cosine(a, b)
        rows_A1.append({"comparison": descr, "cosine": round(sim, 4)})
        print(f"  cos({descr}) = {sim:+.4f}")

    # Random-token baseline: average cosine to a random sample of token directions
    torch.manual_seed(42)
    random_ids = torch.randperm(model.cfg.d_vocab)[:200].tolist()
    country_random = torch.tensor([cosine(wu("country"), model.W_U[:, r])
                                   for r in random_ids])
    capital_random = torch.tensor([cosine(wu("capital"), model.W_U[:, r])
                                   for r in random_ids])
    print(f"  cos({country_str!r:14s} vs random 200 tokens) = "
          f"mean {country_random.mean().item():+.4f}, std {country_random.std().item():.4f}")
    print(f"  cos({capital_str!r:14s} vs random 200 tokens) = "
          f"mean {capital_random.mean().item():+.4f}, std {capital_random.std().item():.4f}")
    rows_A1.append({
        "comparison": f"{country_str!r} vs random (mean+std)",
        "cosine": f"{country_random.mean().item():+.4f} ± {country_random.std().item():.4f}",
    })
    rows_A1.append({
        "comparison": f"{capital_str!r} vs random (mean+std)",
        "cosine": f"{capital_random.mean().item():+.4f} ± {capital_random.std().item():.4f}",
    })

    pd.DataFrame(rows_A1).to_csv(
        f"results_geometry/{label}_unembed_analysis.csv", index=False)

    # =========================================================
    # Run prompt and find top contributing heads if not supplied
    # =========================================================
    toks = model.to_tokens(prompt)
    seq_strs = [model.to_string(t.item()) for t in toks[0]]
    print(f"\n  Prompt tokenization ({len(seq_strs)} tokens):")
    for i, s in enumerate(seq_strs):
        marker = "  <-- country" if s == country_str else ""
        print(f"    {i:2d}: {s!r}{marker}")

    actual_country_positions = [i for i, s in enumerate(seq_strs) if s == country_str]
    print(f"  Country '{country_str}' appears at positions: {actual_country_positions}")

    # Forward + cache
    logits = model(toks)
    final_probs = torch.softmax(logits[0, -1].float(), dim=-1)
    final_top = torch.topk(final_probs, 6)
    print(f"\n  Model output top-6 at final position:")
    for idx, p in zip(final_top.indices, final_top.values):
        print(f"    {repr(model.to_string(idx.item())):14s}  P={p.item():.4f}")

    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("attn_out" in n or n.endswith("attn.hook_z") or "pattern" in n),
    )

    # If no top_heads supplied (Part B case), compute them dynamically:
    # the top heads by |contribution to capital| at the layers whose attn_out
    # contributes most.
    if top_heads is None:
        attn_contribs = []
        for L in range(n_layers):
            attn_out = cache["attn_out", L][0, -1]
            attn_contribs.append((L, abs(project_to_token(attn_out, ids["capital"]))))
        top_attn_layers = sorted([L for L, _ in sorted(
            attn_contribs, key=lambda x: -x[1])[:5]])
        print(f"\n  Top attention-contributing layers: {top_attn_layers}")

        head_contribs = []
        for L in top_attn_layers:
            z = cache["z", L][0, -1]
            W_O_L = model.W_O[L]
            head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
            for h in range(n_heads):
                head_contribs.append((L, h, head_out[h],
                                      project_to_token(head_out[h], ids["capital"])))
        # Top 6 heads by |contribution to capital|
        head_contribs.sort(key=lambda t: -abs(t[3]))
        top_heads = [(L, h) for L, h, _, _ in head_contribs[:6]]
        print(f"  Top 6 contributing heads (computed): {top_heads}")

    # =========================================================
    # A2. Each top head's full top-20 logit-lens projection
    # =========================================================
    print(f"\n[A2] Top-20 logit-lens projection of each top-contributing head's output")
    rows_A2 = []
    for (L, h) in top_heads:
        z = cache["z", L][0, -1]
        W_O_L = model.W_O[L]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
        head_vec = head_out[h]
        top20 = logit_lens_topk(head_vec, k=20)

        # Find positions of country/capital tokens in the top-20
        country_rank = next((i for i, (t, _) in enumerate(top20)
                            if t == country_str), None)
        capital_rank = next((i for i, (t, _) in enumerate(top20)
                            if t == capital_str), None)

        print(f"\n  L{L:02d} head_{h:2d}:")
        print(f"    top-20: {top20[:8]}")
        print(f"    ...        {top20[8:16]}")
        print(f"    ...        {top20[16:]}")
        print(f"    {country_str!r} rank in this head's top-20: "
              f"{country_rank if country_rank is not None else 'not in top-20'}")
        print(f"    {capital_str!r} rank in this head's top-20: "
              f"{capital_rank if capital_rank is not None else 'not in top-20'}")
        rows_A2.append({
            "layer": L, "head": h,
            "top20": str(top20),
            "country_rank": country_rank,
            "capital_rank": capital_rank,
        })
    pd.DataFrame(rows_A2).to_csv(
        f"results_geometry/{label}_head_top20.csv", index=False)

    # =========================================================
    # A3. Decompose each head's output into country/capital/orthogonal
    # =========================================================
    print(f"\n[A3] Decomposition of each head's output along country vs capital direction")
    # Project onto unit-vector versions of W_U[country] and W_U[capital]
    u_country = F.normalize(wu("country").float(), dim=0)
    u_capital = F.normalize(wu("capital").float(), dim=0)

    rows_A3 = []
    for (L, h) in top_heads:
        z = cache["z", L][0, -1]
        W_O_L = model.W_O[L]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
        head_vec = head_out[h].float()

        # Scalar projections
        proj_country = (head_vec @ u_country).item()
        proj_capital = (head_vec @ u_capital).item()
        # Magnitudes
        mag = head_vec.norm().item()
        # Fractions of total magnitude
        frac_country = abs(proj_country) / (mag + 1e-9)
        frac_capital = abs(proj_capital) / (mag + 1e-9)

        rows_A3.append({
            "layer": L, "head": h,
            "head_output_norm": round(mag, 2),
            f"proj_onto_{country_str.strip()}": round(proj_country, 2),
            f"proj_onto_{capital_str.strip()}": round(proj_capital, 2),
            f"frac_{country_str.strip()}": round(frac_country, 4),
            f"frac_{capital_str.strip()}": round(frac_capital, 4),
        })

    df_A3 = pd.DataFrame(rows_A3)
    df_A3.to_csv(f"results_geometry/{label}_head_decomposition.csv", index=False)
    print(df_A3.to_string(index=False))

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


# ============================================================
# PART A — Germany / Berlin
# ============================================================
analyze_pair(
    label="germany",
    country_str=" Germany",
    capital_str=" Berlin",
    other_country_str=" France",
    other_capital_str=" Paris",
    other_same_country_city_str=" Frankfurt",
    prompt="Q: What is the capital of Germany?\nA: The capital of Germany is",
    country_positions=None,  # script discovers them
    top_heads=[(17, 6), (15, 7), (17, 0), (13, 1), (15, 3), (16, 6)],  # from experiment 12
)


# ============================================================
# PART B — California / Sacramento (with fallback)
# ============================================================
print(f"\n\n{'#'*72}")
print("# PART B: Replication on US state -> capital")
print(f"{'#'*72}")

# Candidate pairs in order of preference. The script will use the first
# one where both the state and capital are single tokens.
candidates = [
    {
        "label": "california",
        "state": " California", "capital": " Sacramento",
        "other_state": " Texas", "other_capital": " Austin",
        "other_city": " Hollywood",
        "prompt": "Q: What is the capital of California?\nA: The capital of California is",
    },
    {
        "label": "texas",
        "state": " Texas", "capital": " Austin",
        "other_state": " Florida", "other_capital": " Tallahassee",
        "other_city": " Dallas",
        "prompt": "Q: What is the capital of Texas?\nA: The capital of Texas is",
    },
    {
        "label": "massachusetts",
        "state": " Massachusetts", "capital": " Boston",
        "other_state": " Virginia", "other_capital": " Richmond",
        "other_city": " Cambridge",
        "prompt": "Q: What is the capital of Massachusetts?\nA: The capital of Massachusetts is",
    },
    {
        "label": "ohio",
        "state": " Ohio", "capital": " Columbus",
        "other_state": " Michigan", "other_capital": " Lansing",
        "other_city": " Cleveland",
        "prompt": "Q: What is the capital of Ohio?\nA: The capital of Ohio is",
    },
]

selected = None
for c in candidates:
    needed = [c["state"], c["capital"], c["other_state"],
              c["other_capital"], c["other_city"]]
    if all(safe_id(s) is not None for s in needed):
        selected = c
        print(f"\n  Selected pair: {c['state']!r} -> {c['capital']!r}")
        break
    else:
        bad = [s for s in needed if safe_id(s) is None]
        print(f"  Skipping {c['label']}: not single tokens: {bad}")

if selected is None:
    print("\nFAIL: no candidate state-capital pair has all single-token tokens.")
else:
    analyze_pair(
        label=selected["label"],
        country_str=selected["state"],
        capital_str=selected["capital"],
        other_country_str=selected["other_state"],
        other_capital_str=selected["other_capital"],
        other_same_country_city_str=selected["other_city"],
        prompt=selected["prompt"],
        country_positions=None,
        top_heads=None,  # discover dynamically
    )


print(f"\n{'='*72}")
print("How to read these results:")
print()
print("[A1] Cosine similarities. If cos(country, capital) is SUBSTANTIALLY")
print("     higher than the random-token baseline (mean+1std), the unembedding")
print("     directly encodes the country-capital association as geometric")
print("     proximity. Compare to cos(country, OTHER capital) to ensure the")
print("     association is specific, not just 'all proper nouns are similar'.")
print()
print("[A2] Each top head's full top-20 logit-lens projection. If the country")
print("     ranks higher than (or near) the capital in the head's own top-20,")
print("     the head is writing country-flavored not capital-flavored output,")
print("     confirming the experiment-12 interpretation. If the capital is")
print("     rank 1 cleanly, the head IS specifically writing the capital and")
print("     the unembedding-geometry framing is too strong.")
print()
print("[A3] Decomposition. frac_country > frac_capital means the head's output")
print("     lies more along the country direction than the capital direction.")
print("     If frac_country is consistently ~2-3x frac_capital across heads,")
print("     the 'amplify country, unembedding does the rest' story is confirmed.")
print()
print("Part B parallels: if California shows the same pattern, the mechanism")
print("generalizes. If California is qualitatively different, Germany was a")
print("special case (perhaps because of training-data frequency).")
print(f"{'='*72}")
