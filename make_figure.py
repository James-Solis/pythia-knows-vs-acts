"""
make_figure.py — render the one headline figure for the README.

Reads the color-geometry summary you already generated and plots how many of
the four named-entity circuit heads (L17h6, L15h7, L17h0, L13h1) appear among
the top-6 answer contributors for each color fact. The measured result is 0/4
for every color; the named-entity case is 4/4 by construction (those heads were
identified as the dominant country-capital contributors in experiments 13-14),
so it is drawn as a reference line rather than a measured bar.

Run from the repo root after 15_color_geometry.py has produced its summary:
    python make_figure.py
Output: figures/circuit_engagement.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

COLOR_SUMMARY = "results_color_geometry/summary.csv"
N_CIRCUIT_HEADS = 4  # L17h6, L15h7, L17h0, L13h1

os.makedirs("figures", exist_ok=True)

if not os.path.exists(COLOR_SUMMARY):
    raise SystemExit(
        f"{COLOR_SUMMARY} not found. Run 15_color_geometry.py first to generate it."
    )

df = pd.read_csv(COLOR_SUMMARY)

# Column written by 15_color_geometry.py (one row per color pair).
count_col = "n_circuit_heads_in_top6"
label_col = "label" if "label" in df.columns else df.columns[0]
if count_col not in df.columns:
    raise SystemExit(
        f"Column {count_col!r} not in {COLOR_SUMMARY}. Found: {list(df.columns)}"
    )

labels = df[label_col].astype(str).tolist()
counts = df[count_col].tolist()

fig, ax = plt.subplots(figsize=(6, 4))

bars = ax.bar(labels, counts, color="#4C78A8", width=0.6, zorder=3)
# Named-entity reference: 4/4 by construction.
ax.axhline(N_CIRCUIT_HEADS, color="#E45756", linestyle="--", linewidth=1.6, zorder=2)
ax.text(
    len(labels) - 0.5, N_CIRCUIT_HEADS - 0.18,
    "named-entity facts: 4/4 (by construction)",
    color="#E45756", ha="right", va="top", fontsize=9,
)

for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.08, str(int(c)),
            ha="center", va="bottom", fontsize=10)

ax.set_ylim(0, N_CIRCUIT_HEADS + 0.6)
ax.set_yticks(range(N_CIRCUIT_HEADS + 1))
ax.set_ylabel("Circuit heads in top-6 contributors")
ax.set_xlabel("Color fact")
ax.set_title("The named-entity circuit does not engage for color facts\n"
             "(Pythia-1.4B, four-head circuit L17h6 / L15h7 / L17h0 / L13h1)",
             fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = "figures/circuit_engagement.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
