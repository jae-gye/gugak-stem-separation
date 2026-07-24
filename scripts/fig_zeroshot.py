#!/usr/bin/env python3
"""Zero-shot baseline deliverable figure: per-model energy distribution + 타악 recovery SI-SDR.

Reads experiments/zeroshot_baseline_260719/metrics.parquet -> .../figures/fig_zeroshot_baseline.png.
Note: model output heads (drums/bass/other/vocals) are the Western models' own stem names,
kept as-is. Our gugak group 타악 is kept in Korean (not anglicised); the SI-SDR measures how
well each model's `drums` head recovers ground-truth 타악.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import koreanize_matplotlib  # noqa: E402,F401
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "zeroshot_baseline_260719"
m = pd.read_parquet(EXP_DIR / "metrics.parquet")
MODELS = [("htdemucs", "htdemucs"), ("bsroformer", "4-stem BS-RoFormer")]
STACK = ["drums", "bass", "vocals", "other"]          # model output heads (kept as-is)
SC = {"drums": "#D55E00", "bass": "#0072B2", "vocals": "#009E73", "other": "#BDBDBD"}
MC = {"htdemucs": "#0072B2", "bsroformer": "#E69F00"}

si = m.groupby("genre")[["htdemucs_tak_drums_sisdr", "bsroformer_tak_drums_sisdr"]].median()
order = si.sort_values("htdemucs_tak_drums_sisdr").index.tolist()
y = np.arange(len(order))

fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.15]})

for ax, (mkey, mname) in zip(axes[:2], MODELS):
    eg = m.groupby("genre")[[f"{mkey}_{s}_pct" for s in STACK]].mean().reindex(order)
    left = np.zeros(len(order))
    for s in STACK:
        vals = eg[f"{mkey}_{s}_pct"].values
        ax.barh(y, vals, left=left, color=SC[s], height=0.72, edgecolor="white", linewidth=0.5)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlim(0, 100); ax.set_xlabel("% of predicted energy")
    ax.set_title(f"{mname} — where gugak energy lands", fontsize=12, weight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(handles=[Patch(color=SC[s], label=s) for s in STACK],
               frameon=False, ncol=4, fontsize=9, loc="upper left", bbox_to_anchor=(0.15, -0.13))

ax = axes[2]
h = 0.38
for i, (mkey, mname) in enumerate(MODELS):
    vals = si.loc[order, f"{mkey}_tak_drums_sisdr"].values
    ax.barh(y + (0.5 - i) * h, vals, height=h, color=MC[mkey], label=mname)
    for yi, v in zip(y + (0.5 - i) * h, vals):
        ax.text(v + (0.6 if v >= 0 else -0.6), yi, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
ax.axvline(0, color="#444", lw=1.0)
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlabel("타악 SI-SDR (dB) — higher = better")
ax.set_title("타악 recovery  (via each model's 'drums' head)", fontsize=12, weight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.margins(x=0.18)

fig.suptitle("Zero-shot baseline — Western separators on gugak (91 val songs): "
             "melodic → \"other\"; 타악 recovered only for Western-adjacent 창작국악",
             fontsize=13, weight="bold", y=1.0)
fig.tight_layout()
out = EXP_DIR / "figures" / "fig_zeroshot_baseline.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)


if __name__ == "__main__":
    pass
