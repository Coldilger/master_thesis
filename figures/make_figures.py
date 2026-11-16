"""Rebuilds every figure of the thesis from the numbers reported in the three
model repositories (source file given next to each block).

    python make_figures.py        # writes fig_*.png next to this file

No raw data is read: the numbers are copied from the result documents, so the
figures can be rebuilt even if the cluster's scratch storage is gone.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
C = {"f1": "#3b6fb6", "mimic": "#d9822b", "lda": "#3f9b6b", "base": "#8a8f98", "bad": "#c0392b", "ink": "#22262b"}
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def success_bars(ax, rows, color, title):
    """rows: (label, successes, n, is_baseline)"""
    for i, (label, k, n, is_base) in enumerate(rows):
        p = 100 * k / n
        lo, hi = wilson(k, n)
        ax.bar(i, p, color=C["base"] if is_base else color, width=0.68)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=C["ink"], capsize=3, lw=1)
        ax.text(i, hi + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 75)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def save(fig, name):
    fig.savefig(HERE / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Experiment 1
# F1-VLA eval/bridge/experiment1_ablation/README.md; mimic-video
# eval/bridge/experiment1_ablation/README.md (closed-loop table, rerun
# baseline); LDA-1B eval/robocasa/RESULTS.md (seeded rerun).
def exp1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [4, 6, 2.2]})
    success_bars(axes[0], [("baseline", 140, 288, True), ("tokens\nremoved", 100, 288, False),
                           ("wrong-episode\nforecast", 144, 288, False), ("task-average\nforecast", 139, 288, False)],
                 C["f1"], "F1-VLA (SimplerEnv Bridge, 288 episodes)")
    success_bars(axes[1], [("baseline", 46, 96, True), ("zeros", 0, 96, False), ("task-average\ncontext", 0, 96, False),
                           ("other clip\n(coherent)", 0, 96, False), ("per-step\nshuffle", 0, 96, False),
                           ("episode's first\ncontext, frozen", 0, 96, False)],
                 C["mimic"], "mimic-video (SimplerEnv Bridge, 96 episodes)")
    success_bars(axes[2], [("forecast off\n(default)", 67, 150, True), ("forecast\non", 62, 150, False)],
                 C["lda"], "LDA-1B (RoboCasa, 150)")
    axes[0].set_ylabel("task success, % (Wilson 95% CI)")
    fig.suptitle("Experiment 1 — replace or remove the forecast, measure closed-loop success", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp1_ablation.png")


# ---------------------------------------------------------------- Experiment 2
# Live oracle probes (actions compared at identical live states, never to
# dataset actions): F1-VLA eval/bridge/experiment2_oracle/ORACLE_EXPERIMENT.md
# (job 631445), mimic-video .../ORACLE_EXPERIMENT.md (job 634004, stop=1),
# LDA-1B eval/robocasa/internals/README.md (180 decisions).
def exp2():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        (axes[0], "F1-VLA", C["f1"], [("oracle vs.\nown action", 0.02021), ("size of the\naction itself", 0.11030)], "action L1"),
        (axes[1], "mimic-video (k=1)", C["mimic"], [("oracle vs.\nown action", 0.03080), ("zero action vs.\nown action", 0.28982)], "action L1"),
    ]
    for ax, title, col, rows, yl in panels:
        ax.bar([0, 1], [r[1] for r in rows], color=[col, C["base"]], width=0.6)
        for i, r in enumerate(rows):
            ax.text(i, r[1], f"{r[1]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], [r[0] for r in rows])
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    ax = axes[2]
    rows = [("real next\nframe", 0.29), ("zeros", 0.06), ("noise", 0.11), ("other\nepisode", 0.12), ("consecutive\ndecisions", 133.0)]
    ax.bar(range(5), [r[1] for r in rows], color=[C["lda"]] * 4 + [C["base"]], width=0.65)
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 1.15, f"{r[1]:g}%", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 600)
    ax.set_xticks(range(5), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("change of the action chunk, % (log)")
    ax.set_title("LDA-1B: forecast slot replaced by …")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Experiment 2 — give the model the real future: how far does the action move?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp2_oracle.png")


# ---------------------------------------------------------------- Experiment 3
# experiment3_cost/README.md of each repository (median per decision).
def exp3():
    rows = [("F1-VLA, forecast removed", 160.6, C["base"]), ("F1-VLA (deployed)", 215.7, C["f1"]),
            ("LDA-1B (deployed, no forecast)", 254.7, C["lda"]), ("LDA-1B, forecast on", 428.2, C["lda"]),
            ("mimic-video, k=1 (deployed)", 1060.0, C["mimic"]), ("mimic-video, k=23", 10226.9, C["mimic"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    for i, (label, v, col) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.62)
        ax.text(v * 1.08, i, f"{v:,.0f} ms", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(80, 40000)
    ax.set_xlabel("median latency per decision, ms (log scale)")
    ax.set_title("Experiment 3 — cost of one decision")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_exp3_cost.png")


# ---------------------------------------------------------------- Experiment 4
# experiment4_probing/README.md of each repository: gain of a probe over the
# best constant predictor on held-out episodes (ridge; MLP shown for LDA-1B).
def exp4():
    labels = ["F1-VLA\nridge", "mimic-video\nridge", "LDA-1B\nridge", "LDA-1B\nMLP"]
    current = [-254.2, -251.0, -229.0, 59.1]
    future = [-19.3, 0.6, 40.6, 65.7]
    cols = [C["f1"], C["mimic"], C["lda"], C["lda"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    w = 0.38
    for i in range(4):
        ax.bar(i - w / 2, current[i], w, color=cols[i])
        ax.bar(i + w / 2, future[i], w, color=cols[i], alpha=0.45, hatch="//", edgecolor="white")
        for x, v in ((i - w / 2, current[i]), (i + w / 2, future[i])):
            ax.text(x, v + (6 if v >= 0 else -6), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(range(4), labels)
    ax.set_ylim(-300, 110)
    ax.set_ylabel("gain over a constant predictor, %")
    ax.bar(0, 0, color="#555", label="current pose")
    ax.bar(0, 0, color="#555", alpha=0.45, hatch="//", edgecolor="white", label="future pose (5 steps ahead)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Experiment 4 — can the end-effector pose be read from the hidden state?")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_exp4_probing.png")


# ---------------------------------------------------------------- Experiment 5
# experiment5_erasure/README.md of each repository (LEACE).
def exp5():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7))
    ax = axes[0]
    names, cols = ["F1-VLA", "mimic-video", "LDA-1B"], [C["f1"], C["mimic"], C["lda"]]
    before, after = [100.0, 100.0, 98.7], [1.3, 2.7, 55.3]
    w = 0.38
    for i in range(3):
        ax.bar(i - w / 2, before[i], w, color=cols[i])
        ax.bar(i + w / 2, after[i], w, color=cols[i], alpha=0.45, hatch="//", edgecolor="white")
        ax.text(i - w / 2, before[i] + 2, f"{before[i]:.0f}%", ha="center", fontsize=8.5)
        ax.text(i + w / 2, after[i] + 2, f"{after[i]:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0, 118)
    ax.set_ylabel("episode-identity accuracy, %")
    ax.set_title("Episode identity: before / after erasure (hatched)")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    rows = [("F1-VLA\nridge", -254.2, -257.0, C["f1"]), ("mimic-video\nridge", -251.4, -250.0, C["mimic"]),
            ("LDA-1B MLP\nepisode erased", 59.1, 56.9, C["lda"]), ("LDA-1B MLP\npose direction erased", 59.1, -11.4, C["lda"])]
    for i, (label, b, a, col) in enumerate(rows):
        ax.bar(i - w / 2, b, w, color=col)
        ax.bar(i + w / 2, a, w, color=col, alpha=0.45, hatch="//", edgecolor="white")
        for x, v in ((i - w / 2, b), (i + w / 2, a)):
            ax.text(x, v + (6 if v >= 0 else -6), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(range(4), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-300, 110)
    ax.set_ylabel("current-pose gain over constant, %")
    ax.set_title("Pose probe: before / after erasure (hatched)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Experiment 5 — erase a concept from the hidden state, probe again", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp5_erasure.png")


# ------------------------------------------------------- LDA-1B camera viewpoint
# LDA-1B eval/robocasa/RESULTS.md, 2026-08-18.
def camera():
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    success_bars(ax, [("egocentric\n(as trained)", 24, 50, True), ("third-person", 0, 50, False)], C["lda"],
                 "LDA-1B on RoboCasa: camera viewpoint")
    ax.set_ylabel("task success, % (Wilson 95% CI)")
    save(fig, "fig_lda_camera_viewpoint.png")


# ---------------------------------------------------------------- Experiment 9
# mimic-video eval/bridge/experiment9_depth/README.md.
K_OL = [0, 1, 2, 4, 8, 16, 23, 30]
FOLLOW_FUTURE = [0.77, 0.79, 0.82, 0.87, 0.85, 0.84, 0.80, 0.76]
FOLLOW_OBS = [0.23, 0.22, 0.20, 0.13, 0.17, 0.18, 0.21, 0.26]
FUTURE_SPREAD = [9, 11, 15, 31, 50, 59, 61, 63]
ACTION_SPREAD = [3, 4, 6, 14, 19, 32, 32, 21]
FORECAST_ERR = [0.69, 0.69, 0.70, 0.74, 0.86, 0.95, 0.97, 0.99]
K_CL = [0, 1, 2, 4, 8, 16]
LAT = [0.82, 1.36, 1.89, 2.95, 5.09, 9.30]
SUCC = [43, 46, 47, 41, 30, 9]


def exp9_open():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = range(len(K_OL))
    ax = axes[0]
    ax.plot(x, FOLLOW_FUTURE, "o-", color=C["mimic"], label="future tokens swapped")
    ax.plot(x, FOLLOW_OBS, "s--", color=C["base"], label="observed tokens swapped (control)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("how far the action follows the other seed (0–1)")
    ax.set_title("The decoder follows the imagined future")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax = axes[1]
    ax.plot(x, FUTURE_SPREAD, "o-", color=C["mimic"], label="imagined futures")
    ax.plot(x, ACTION_SPREAD, "s-", color=C["ink"], label="actions")
    ax.set_ylabel("difference across video seeds, %")
    ax.set_title("With depth, futures become seed-specific")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    ax.plot(x, FORECAST_ERR, "o-", color=C["bad"])
    ax.axhline(1.0, color=C["base"], ls="--", lw=1)
    ax.text(0, 1.005, "static scene", color=C["base"], fontsize=8.5, va="bottom")
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("forecast error / error of a static scene")
    ax.set_title("Deeper forecasts are not closer to the truth")
    for ax in axes:
        ax.set_xticks(list(x), K_OL)
        ax.set_xlabel("denoising depth k (of 35; deployed k=1)")
        ax.axvline(1, color=C["mimic"], alpha=0.18, lw=8)
    fig.suptitle("Experiment 9, open loop — mimic-video, 16 real Bridge clips", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp9_open_loop.png")


def exp9_closed(extra_replan=None):
    """extra_replan: optional list of (label, successes, n) appended to the replanning panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    p = [100 * s / 96 for s in SUCC]
    ci = [wilson(s, 96) for s in SUCC]
    ax.errorbar(LAT, p, yerr=[[a - c[0] for a, c in zip(p, ci)], [c[1] - a for a, c in zip(p, ci)]],
                fmt="o-", color=C["mimic"], capsize=3, label="one video seed")
    for k, l, v in zip(K_CL, LAT, p):
        ax.annotate(f"k={k}", (l, v), textcoords="offset points", xytext=(6, 7), fontsize=8.5)
    lo, hi = wilson(47, 96)
    v = 100 * 47 / 96
    ax.errorbar([4 * 9.30], [v], yerr=[[v - lo], [hi - v]], fmt="D", color=C["ink"], capsize=3,
                label="k=16, action averaged over 4 video seeds")
    ax.annotate("k=16 × 4 seeds", (4 * 9.30, v), textcoords="offset points", xytext=(-78, 9), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.set_xlabel("latency per decision, s (log; open-loop measurement, H100)")
    ax.set_ylabel("task success, % (Wilson 95% CI)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title("Success vs. latency: more generation does not buy success")
    rows = [("1", 0, 96, False), ("2", 0, 96, False), ("5\n(deployed)", 46, 96, True), ("10", 38, 96, False)]
    for label, k, n in (extra_replan or []):
        rows.append((label, k, n, False))
    success_bars(axes[1], rows, C["mimic"], "Replanning rate at k=1")
    axes[1].set_xlabel("actions executed per decision (of 15 planned)")
    fig.suptitle("Experiment 9, closed loop — mimic-video, 96 SimplerEnv Bridge episodes", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp9_closed_loop.png")


# -------------------------------------------------------------------- overview
def overview():
    """One row per model: deployed success and the cleanest content-removing test."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    rows = [("F1-VLA", C["f1"], 140, 288, 139, 288, "task-average forecast"),
            ("mimic-video", C["mimic"], 46, 96, 0, 96, "task-average context"),
            ("LDA-1B", C["lda"], 67, 150, 62, 150, "forecast switched on")]
    w = 0.38
    for i, (name, col, k0, n0, k1, n1, what) in enumerate(rows):
        for x, k, n, kw in ((i - w / 2, k0, n0, {}), (i + w / 2, k1, n1, {"alpha": 0.45, "hatch": "//", "edgecolor": "white"})):
            v = 100 * k / n
            lo, hi = wilson(k, n)
            ax.bar(x, v, w, color=col, **kw)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=C["ink"], capsize=3, lw=1)
            ax.text(x, hi + 2, f"{k}/{n}", ha="center", fontsize=8.5)
    ax.set_xticks(range(3), [f"{r[0]}\nhatched:\n{r[6]}" for r in rows])
    ax.set_ylim(0, 72)
    ax.set_ylabel("task success, % (Wilson 95% CI)")
    ax.bar(0, 0, color="#555", label="deployed policy")
    ax.bar(0, 0, color="#555", alpha=0.45, hatch="//", edgecolor="white", label="forecast changed (see axis label)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Does changing the forecast change the outcome?")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_overview.png")


# ---------------------------------------------------------------- Experiment 8
# F1-VLA eval/bridge/experiment8_conflict/README.md (steps 0-1, mirror
# instruction, 1785 decisions; mean over episodes with 95% bootstrap interval)
def exp8():
    tasks = ["Carrot", "Eggplant", "Spoon", "Stack"]
    tok_seed = [(0.74, 0.71, 0.77), (0.67, 0.62, 0.72), (0.74, 0.71, 0.76), (0.71, 0.68, 0.74)]
    tok_instr = [(0.19, 0.17, 0.21), (0.17, 0.15, 0.19), (0.22, 0.20, 0.24), (0.19, 0.18, 0.21)]
    dist = {  # (mean, lo, hi) per task
        "instruction changed everywhere: d(BB, AA)": ([(1.971, 1.761, 2.178), (1.370, 1.265, 1.477), (2.003, 1.805, 2.204), (1.589, 1.381, 1.805)], C["base"]),
        "whole generation block from B: d(AB, AA)": ([(0.0053, 0.0046, 0.0063), (0.0052, 0.0045, 0.0060), (0.0043, 0.0039, 0.0047), (0.0035, 0.0033, 0.0037)], C["f1"]),
        "only forecast positions from B: d(ABf, AA)": ([(0.0064, 0.0039, 0.0099), (0.0046, 0.0036, 0.0060), (0.0034, 0.0030, 0.0039), (0.0029, 0.0028, 0.0031)], "#8fb0dc"),
        "noise floor, other forecast seed: d(AA2, AA)": ([(0.0048, 0.0038, 0.0063), (0.0046, 0.0036, 0.0061), (0.0034, 0.0031, 0.0037), (0.0029, 0.0027, 0.0031)], C["ink"]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    w = 0.36
    for j, (vals, color, label) in enumerate([(tok_seed, C["ink"], "other sampling seed, same instruction"),
                                               (tok_instr, C["f1"], "other instruction, same seed")]):
        x = [i + (j - 0.5) * w for i in range(4)]
        m = [100 * v[0] for v in vals]
        ax.bar(x, m, width=w, color=color, label=label)
        ax.errorbar(x, m, yerr=[[100 * (v[0] - v[1]) for v in vals], [100 * (v[2] - v[0]) for v in vals]],
                    fmt="none", color="#666b73", capsize=2.5, lw=1)
    ax.set_xticks(range(4), tasks)
    ax.set_ylim(0, 100)
    ax.set_ylabel("forecast tokens that change, %")
    ax.set_title("The instruction changes the forecast\nless than resampling it does")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    w = 0.2
    for j, (label, (vals, color)) in enumerate(dist.items()):
        x = [i + (j - 1.5) * w for i in range(4)]
        m = [v[0] for v in vals]
        ax.bar(x, m, width=w, color=color, label=label)
        ax.errorbar(x, m, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="none", color="#666b73", capsize=2, lw=1)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 30)
    ax.set_xticks(range(4), tasks)
    ax.set_ylabel("change of the action chunk (log scale)")
    ax.set_title("The action follows the instruction;\na forecast made for another instruction moves it no more than noise")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Experiment 8, single decisions — F1-VLA, mirrored instruction, 1785 decisions on SimplerEnv Bridge",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_exp8_conflict.png")


if __name__ == "__main__":
    overview(); exp1(); exp2(); exp3(); exp4(); exp5(); camera(); exp9_open()
    exp9_closed(extra_replan=[("1, plan\nstep 5", 17, 96)]); exp8()
