"""
Experiment 15: Does the country-capital circuit work for color attributes?

Applies the same three-measurement analysis from experiments 13-14 to three
color-attribute facts: sky, grass, banana.

Prior context that shapes the question:
  - Addendum 5's full survey showed colors do NOT show the layer-16 dip
    in corrected runs that named entities do (sky=green delta L15->L16
    was +0.02; named entities were -0.32 and -0.23).
  - The associative-expansion mechanism that defined the country-capital
    circuit didn't engage for colors because the neighborhood is shallow.
  - The first addendum dropped banana as a diagnostic because the model
    represents banana color as VARIABLE rather than canonical.

This experiment asks: does the four-head circuit (L17 head_6, L15 head_7,
L17 head_0, L13 head_1) engage for color attributes? Or do colors use a
different mechanism entirely?

Three pairs:
  sky -> blue   (canonical color attribute)
  grass -> green (canonical color attribute)
  banana -> yellow (the model represents this as variable; expected weak/null)

Same three measurements per pair:
  [A1] Cosine similarity in W_U between topic and color, plus baselines
  [A2] Top-contributing heads' top-20 logit-lens projections
  [A3] Decomposition of each head's output along topic vs color direction

Plus one extra measurement specific to colors:
  [A4] Is the country-capital circuit's four specific heads also top
       contributors for colors? Or are different heads doing the work?
       This is the core question for whether the circuit is general.

Output:
  results_color_geometry/<fact>_*.csv
  results_color_geometry/summary.csv
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

os.makedirs("results_color_geometry", exist_ok=True)

model = get_model()
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

# The four heads identified as the country-capital circuit
CIRCUIT_HEADS = [(17, 6), (15, 7), (17, 0), (13, 1)]


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


# Random-token baseline for cosine — computed once and reused
torch.manual_seed(42)
RANDOM_IDS = torch.randperm(model.cfg.d_vocab)[:200].tolist()


def random_cosine_stats(target_vec):
    sims = torch.tensor([cosine(target_vec, model.W_U[:, r]) for r in RANDOM_IDS])
    return sims.mean().item(), sims.std().item()


# Three color-attribute pairs
PAIRS = [
    {
        "label": "sky_blue",
        "topic": " sky",
        "answer": " blue",
        "other_topic": " grass",
        "other_answer": " green",
        "same_neighborhood": " white",   # alternative-but-related color attribute (clouds, snow)
        "prompt": "Q: What color is the sky?\nA: The sky is",
    },
    {
        "label": "grass_green",
        "topic": " grass",
        "answer": " green",
        "other_topic": " sky",
        "other_answer": " blue",
        "same_neighborhood": " yellow",   # alt color seen on dry/dead grass
        "prompt": "Q: What color is grass?\nA: Grass is",
    },
    {
        "label": "banana_yellow",
        "topic": " banana",
        "answer": " yellow",
        "other_topic": " apple",
        "other_answer": " red",
        "same_neighborhood": " green",   # unripe banana
        "prompt": "Q: What color is a banana?\nA: A banana is",
    },
]


def analyze_pair(pair):
    topic, answer = pair["topic"], pair["answer"]
    label = pair["label"]

    print(f"\n{'#'*72}")
    print(f"# COLOR PAIR: {topic!r} -> {answer!r}")
    print(f"{'#'*72}")

    # Resolve all tokens
    ids = {}
    for tag, s in [("topic", topic), ("answer", answer),
                   ("other_topic", pair["other_topic"]),
                   ("other_answer", pair["other_answer"]),
                   ("same_neighborhood", pair["same_neighborhood"])]:
        tid = safe_id(s)
        if tid is None:
            print(f"  ERROR: {s!r} is not a clean single token. Aborting this pair.")
            return None
        ids[tag] = tid

    print(f"  topic={topic!r:14s} id={ids['topic']}")
    print(f"  answer={answer!r:14s} id={ids['answer']}")
    print(f"  other_topic={pair['other_topic']!r:14s} id={ids['other_topic']}")
    print(f"  other_answer={pair['other_answer']!r:14s} id={ids['other_answer']}")
    print(f"  same_neighborhood={pair['same_neighborhood']!r:14s} id={ids['same_neighborhood']}")

    # [A1] Cosine
    wu_topic = model.W_U[:, ids["topic"]]
    wu_answer = model.W_U[:, ids["answer"]]
    cos_pair = cosine(wu_topic, wu_answer)
    rand_mean_t, rand_std_t = random_cosine_stats(wu_topic)
    rand_mean_a, rand_std_a = random_cosine_stats(wu_answer)

    sigma_t = (cos_pair - rand_mean_t) / rand_std_t
    sigma_a = (cos_pair - rand_mean_a) / rand_std_a

    print(f"\n[A1] cos({topic!r}, {answer!r}) = {cos_pair:+.4f}")
    print(f"  topic vs random:    mean {rand_mean_t:+.4f} +/- {rand_std_t:.4f}  ({sigma_t:+.1f} sigma)")
    print(f"  answer vs random:   mean {rand_mean_a:+.4f} +/- {rand_std_a:.4f}  ({sigma_a:+.1f} sigma)")
    print(f"  topic vs other_topic ({pair['other_topic']!r}):  {cosine(wu_topic, model.W_U[:, ids['other_topic']]):+.4f}")
    print(f"  answer vs other_answer ({pair['other_answer']!r}): {cosine(wu_answer, model.W_U[:, ids['other_answer']]):+.4f}")
    print(f"  topic vs other_color ({pair['same_neighborhood']!r}):  {cosine(wu_topic, model.W_U[:, ids['same_neighborhood']]):+.4f}")

    # Run the prompt
    toks = model.to_tokens(pair["prompt"])
    seq_strs = [model.to_string(t.item()) for t in toks[0]]
    print(f"\n  Prompt tokenization ({len(seq_strs)} tokens):")
    for i, s in enumerate(seq_strs):
        marker = "  <-- topic" if s == topic else ""
        print(f"    {i:2d}: {s!r}{marker}")

    logits = model(toks)
    final_probs = torch.softmax(logits[0, -1].float(), dim=-1)
    final_top = torch.topk(final_probs, 6)
    print(f"\n  Model output top-6:")
    top1_id = int(final_top.indices[0].item())
    for idx, p in zip(final_top.indices, final_top.values):
        marker = "  <-- canonical answer" if idx.item() == ids["answer"] else ""
        print(f"    {repr(model.to_string(idx.item())):16s} P={p.item():.4f}{marker}")

    p_answer = final_probs[ids["answer"]].item()
    answer_rank = (final_probs > final_probs[ids["answer"]]).sum().item()

    # Cache for head decomposition
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: ("attn_out" in n or n.endswith("attn.hook_z")),
    )

    # Find top 5 attention layers by |contribution to answer|
    layer_contribs = []
    for L in range(n_layers):
        attn_out = cache["attn_out", L][0, -1]
        layer_contribs.append((L, abs(project_to_token(attn_out, ids["answer"]))))
    top_attn_layers = sorted([L for L, _ in sorted(layer_contribs, key=lambda x: -x[1])[:5]])

    # Top 6 heads across those layers
    head_data = []
    for L in top_attn_layers:
        z = cache["z", L][0, -1]
        W_O_L = model.W_O[L]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
        for h in range(n_heads):
            head_data.append((L, h, head_out[h].float(),
                              project_to_token(head_out[h], ids["answer"])))
    head_data.sort(key=lambda t: -abs(t[3]))
    top_heads = head_data[:6]

    # [A2] Each top head's top-20 + topic/answer rank
    print(f"\n[A2] Top-6 contributing heads' rank for {topic!r} vs {answer!r}:")
    head_rank_rows = []
    for L, h, vec, contrib in top_heads:
        top20 = logit_lens_topk(vec, k=20)
        t_rank = next((i for i, (tk, _) in enumerate(top20) if tk == topic), None)
        a_rank = next((i for i, (tk, _) in enumerate(top20) if tk == answer), None)
        in_circuit = "  [CIRCUIT HEAD]" if (L, h) in CIRCUIT_HEADS else ""
        print(f"    L{L:02d} head_{h:2d}  contrib={contrib:+.2f}  "
              f"topic_rank={t_rank}  answer_rank={a_rank}{in_circuit}")
        print(f"        head_top3={top20[:3]}")
        head_rank_rows.append({
            "layer": L, "head": h, "contrib_to_answer": round(contrib, 2),
            "topic_rank_in_top20": t_rank,
            "answer_rank_in_top20": a_rank,
            "is_circuit_head": (L, h) in CIRCUIT_HEADS,
            "head_top3": str(top20[:3]),
        })

    # [A3] Decomposition
    u_topic = F.normalize(wu_topic.float(), dim=0)
    u_answer = F.normalize(wu_answer.float(), dim=0)
    decomp_rows = []
    for L, h, vec, contrib in top_heads:
        proj_t = (vec @ u_topic).item()
        proj_a = (vec @ u_answer).item()
        mag = vec.norm().item()
        decomp_rows.append({
            "layer": L, "head": h,
            "is_circuit_head": (L, h) in CIRCUIT_HEADS,
            "head_output_norm": round(mag, 2),
            "proj_onto_topic": round(proj_t, 2),
            "proj_onto_answer": round(proj_a, 2),
            "frac_topic": round(abs(proj_t) / (mag + 1e-9), 4),
            "frac_answer": round(abs(proj_a) / (mag + 1e-9), 4),
        })
    df_decomp = pd.DataFrame(decomp_rows)

    mean_frac_topic = df_decomp["frac_topic"].mean()
    mean_frac_answer = df_decomp["frac_answer"].mean()
    print(f"\n  Mean across top-6 heads: frac_topic={mean_frac_topic:.3f}  "
          f"frac_answer={mean_frac_answer:.3f}  "
          f"ratio={mean_frac_topic/(mean_frac_answer+1e-9):.2f}")

    # [A4] Specifically check the country-capital circuit heads
    print(f"\n[A4] Country-capital circuit heads' contribution to {answer!r}:")
    circuit_rows = []
    for (L, h) in CIRCUIT_HEADS:
        z = cache["z", L][0, -1]
        W_O_L = model.W_O[L]
        head_out = torch.einsum("hd,hdm->hm", z.float(), W_O_L.float())
        vec = head_out[h].float()
        contrib = project_to_token(vec, ids["answer"])
        top20 = logit_lens_topk(vec, k=20)
        t_rank = next((i for i, (tk, _) in enumerate(top20) if tk == topic), None)
        a_rank = next((i for i, (tk, _) in enumerate(top20) if tk == answer), None)
        proj_t = (vec @ u_topic).item()
        proj_a = (vec @ u_answer).item()
        in_top6 = (L, h) in [(LL, hh) for LL, hh, _, _ in top_heads]

        marker = "  [in top-6]" if in_top6 else ""
        print(f"    L{L:02d} head_{h:2d}  contrib={contrib:+.2f}  "
              f"topic_rank={t_rank}  answer_rank={a_rank}  "
              f"frac_topic={abs(proj_t)/(vec.norm().item()+1e-9):.3f}  "
              f"frac_answer={abs(proj_a)/(vec.norm().item()+1e-9):.3f}{marker}")
        print(f"        head_top3={top20[:3]}")
        circuit_rows.append({
            "layer": L, "head": h,
            "contrib_to_answer": round(contrib, 2),
            "topic_rank_in_top20": t_rank,
            "answer_rank_in_top20": a_rank,
            "frac_topic": round(abs(proj_t) / (vec.norm().item() + 1e-9), 4),
            "frac_answer": round(abs(proj_a) / (vec.norm().item() + 1e-9), 4),
            "in_top6_for_this_fact": in_top6,
            "head_top3": str(top20[:3]),
        })

    # Save
    pd.DataFrame(head_rank_rows).to_csv(
        f"results_color_geometry/{label}_head_ranks.csv", index=False)
    df_decomp.to_csv(
        f"results_color_geometry/{label}_decomposition.csv", index=False)
    pd.DataFrame(circuit_rows).to_csv(
        f"results_color_geometry/{label}_circuit_heads.csv", index=False)

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_circuit_in_top6 = sum(1 for (L, h) in CIRCUIT_HEADS
                            if (L, h) in [(LL, hh) for LL, hh, _, _ in top_heads])

    return {
        "label": label,
        "topic": topic,
        "answer": answer,
        "cosine_topic_answer": round(cos_pair, 4),
        "rand_mean_topic": round(rand_mean_t, 4),
        "sigma_above_random": round(sigma_t, 1),
        "top1_token": repr(model.to_string(top1_id)),
        "top1_is_canonical_answer": (top1_id == ids["answer"]),
        "P(answer)": round(p_answer, 4),
        "answer_rank_at_output": answer_rank,
        "mean_frac_topic": round(mean_frac_topic, 4),
        "mean_frac_answer": round(mean_frac_answer, 4),
        "frac_ratio": round(mean_frac_topic / (mean_frac_answer + 1e-9), 3),
        "n_circuit_heads_in_top6": n_circuit_in_top6,
    }


summary_rows = []
for pair in PAIRS:
    result = analyze_pair(pair)
    if result is not None:
        summary_rows.append(result)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results_color_geometry/summary.csv", index=False)

print(f"\n\n{'='*72}")
print("CROSS-PAIR SUMMARY")
print(f"{'='*72}")
print(summary_df.to_string(index=False))

print(f"\n{'='*72}")
print("Compare to country-capital results:")
print("  Country-capital mean cosine (six pairs): ~0.47")
print("  Country-capital mean P(capital) at output: ~0.59")
print("  Country-capital mean frac_country/frac_capital ratio: ~1.20")
print("  Country-capital: circuit heads (L17h6, L15h7, L17h0, L13h1) all in top-6")
print()
print("Interpretation:")
print("  If cosines, P(answer), and circuit-head overlap are SIMILAR to")
print("  country-capital, the four-head circuit is a general topic-amplifier")
print("  used by colors too.")
print()
print("  If cosines are weak or circuit heads are NOT in the top-6 for colors,")
print("  colors use a different mechanism. The four-head circuit is")
print("  named-entity-specific, not general.")
print()
print("  Banana is the expected odd one out: addendum 2 found the model")
print("  represents banana color as variable, not canonical. Expect weaker")
print("  geometric encoding and a less confident output.")
print(f"{'='*72}")
