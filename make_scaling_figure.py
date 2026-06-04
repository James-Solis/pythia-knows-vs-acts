"""
make_scaling_figure.py — the headline redundancy-vs-scale figure.

Reads results_knockout/<model>/knockout.csv for each model and renders a
two-panel figure:
  (left)  absolute heads needed to break the capital (dethrone + halve) vs scale
  (right) heads-to-dethrone as a FRACTION of the model's attention heads vs scale

Together these show the core finding: the absolute number of heads carrying the
capital grows with scale, while the fraction stays roughly constant (~1%).

Pairs that never dethroned within kmax (e.g. Japan on 2.8b) are excluded from
the mean and the point is drawn as a LOWER BOUND (up-arrow), so the figure does
not overstate.

Run from the repo root after the knockout runs:
    python make_scaling_figure.py
    python make_scaling_figure.py --models pythia-1b pythia-1.4b pythia-2.8b
Output: figures/scaling_redundancy.png
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Pythia parameter counts (billions) and total attention heads (layers * heads)
PARAMS_B = {
    "pythia-410m": 0.41, "pythia-1b": 1.0, "pythia-1.4b": 1.4,
    "pythia-2.8b": 2.8, "pythia-6.9b": 6.9, "pythia-12b": 12.0,
}
TOTAL_HEADS = {
    "pythia-410m": 24 * 16, "pythia-1b": 16 * 8, "pythia-1.4b": 24 * 16,
    "pythia-2.8b": 32 * 32, "pythia-6.9b": 32 * 32, "pythia-12b": 36 * 40,
}

parser = argparse.ArgumentParser(description="Plot the redundancy-vs-scale curve.")
parser.add_argument("--models", nargs="+",
                    default=["pythia-1b", "pythia-1.4b", "pythia-2.8b"])
args = parser.parse_args()

os.makedirs("figures", exist_ok=True)


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


points = []  # one dict per model
for m in args.models:
    path = f"results_knockout/{m}/knockout.csv"
    if not os.path.exists(path):
        print(f"  skip {m}: {path} not found")
        continue
    df = pd.read_csv(path)
    df["dethroned_within_kmax"] = as_bool(df["dethroned_within_kmax"])
    df["halved_within_kmax"] = as_bool(df["halved_within_kmax"])

    dethroned = df[df["dethroned_within_kmax"]]
    halved = df[df["halved_within_kmax"]]
    n_total = len(df)
    n_dethroned = len(dethroned)

    points.append({
        "model": m,
        "params": PARAMS_B.get(m),
        "total_heads": TOTAL_HEADS.get(m),
        "mean_dethrone": dethroned["heads_to_dethrone"].mean() if n_dethroned else float("nan"),
        "mean_halve": halved["heads_to_halve"].mean() if len(halved) else float("nan"),
        "censored": n_dethroned < n_total,           # some pair never fell
        "n_dethroned": n_dethroned, "n_total": n_total,
        "pair_dethrone": list(zip(df["heads_to_dethrone"], df["dethroned_within_kmax"])),
    })

if not points:
    raise SystemExit("No knockout.csv files found. Run experiment 17 first.")

points.sort(key=lambda p: p["params"])
xs = [p["params"] for p in points]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---- Left panel: absolute heads to break the capital ----
deth = [p["mean_dethrone"] for p in points]
halv = [p["mean_halve"] for p in points]

# per-pair scatter for the dethrone metric (shows spread; censored = open up-arrow)
for p in points:
    for val, dethroned in p["pair_dethrone"]:
        if dethroned:
            axL.scatter(p["params"], val, s=18, color="#4C78A8", alpha=0.30, zorder=2)
        else:
            axL.scatter(p["params"], val, s=40, facecolors="none",
                        edgecolors="#E45756", marker="^", zorder=3)

axL.plot(xs, deth, "-o", color="#4C78A8", lw=2, zorder=4, label="heads to dethrone (mean)")
axL.plot(xs, halv, "--s", color="#72B7B2", lw=2, zorder=4, label="heads to halve P (mean)")
for p in points:
    note = f"{p['mean_dethrone']:.1f}" + ("≥" if p["censored"] else "")
    axL.annotate(note, (p["params"], p["mean_dethrone"]),
                 textcoords="offset points", xytext=(6, 6), fontsize=9, color="#4C78A8")

axL.set_xscale("log")
axL.set_xticks(xs)
axL.set_xticklabels([p["model"].replace("pythia-", "") for p in points])
axL.minorticks_off()
axL.set_xlabel("model size (parameters)")
axL.set_ylabel("attention heads removed")
axL.set_title("Absolute heads to break the capital")
axL.set_ylim(bottom=0)
axL.grid(alpha=0.25, zorder=0)
axL.legend(fontsize=9, frameon=False)
axL.spines[["top", "right"]].set_visible(False)

# ---- Right panel: dethrone heads as a fraction of total heads ----
frac = [100.0 * p["mean_dethrone"] / p["total_heads"] for p in points]
axR.plot(xs, frac, "-o", color="#B279A2", lw=2, zorder=4)
for p, f in zip(points, frac):
    note = f"{f:.1f}%" + ("≥" if p["censored"] else "")
    axR.annotate(note, (p["params"], f), textcoords="offset points",
                 xytext=(6, 6), fontsize=9, color="#B279A2")

axR.set_xscale("log")
axR.set_xticks(xs)
axR.set_xticklabels([p["model"].replace("pythia-", "") for p in points])
axR.minorticks_off()
axR.set_xlabel("model size (parameters)")
axR.set_ylabel("% of all attention heads")
axR.set_title("As a fraction of available heads")
axR.set_ylim(0, max(frac) * 1.6)
axR.grid(alpha=0.25, zorder=0)
axR.spines[["top", "right"]].set_visible(False)

fig.suptitle("Redundancy of the capital computation vs. model scale (Pythia)",
             fontsize=12, y=1.02)
fig.text(0.5, -0.04,
         "Greedy attention-head knockout, 6 country-capital pairs. ^ = pair never dethroned "
         "within kmax (lower bound). MLPs not ablated.",
         ha="center", fontsize=8, color="gray")

fig.tight_layout()
out = "figures/scaling_redundancy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Wrote {out}")
for p in points:
    print(f"  {p['model']:13s} dethrone={p['mean_dethrone']:.1f}"
          f"{'≥' if p['censored'] else ' '}  "
          f"({100*p['mean_dethrone']/p['total_heads']:.1f}% of {p['total_heads']} heads, "
          f"{p['n_dethroned']}/{p['n_total']} pairs fell)")
