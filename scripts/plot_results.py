"""Regenerate README figures and statistics from archived, committed Walker results.

Run from any directory: python scripts/plot_results.py
Requires matplotlib==3.10.6. Does not load checkpoints or run training.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def read(path):
        data = (root / path).read_bytes()
        provenance[path] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    def read_json(path):
        return json.loads(read(path))

    curves, manifests = {}, {}
    for seed in (100, 101, 102):
        directory = "pilot" if seed == 100 else "confirmation"
        suffix = "walker-2026-09-20" if seed == 100 else f"walker-s{seed}-2026-09-21"
        base = f"results/m9/{directory}"
        rows = [json.loads(line) for line in read(f"{base}/evaluations-{suffix}.jsonl").splitlines() if line.strip()]
        # Older pilot rows lack segment; after resumes, keep the newest segment.
        surviving = {}
        for row in rows:
            key = row["env_step"]
            if key not in surviving or (row.get("segment") or 0) >= (surviving[key].get("segment") or 0):
                surviving[key] = row
        curves[seed] = [surviving[key] for key in sorted(surviving)]
        manifests[seed] = read_json(f"{base}/run-manifest-{suffix}.json")

    floor = read_json("results/m1/random-floor-walker-2026-09-20.json")
    controls = read_json("results/m9/controls/controls-report-walker-2026-09-20.json")
    floor_mean = mean(floor["returns"])
    final_returns = [mean(curves[seed][-1]["returns"]) for seed in curves]
    stats = {
        "source_commit": "c487336e8d0842d21d8cc37d24e134628a130219",
        "source_sha256": provenance,
        "headline_seed100": {
            "first_five_evaluations_mean": mean(mean(r["returns"]) for r in curves[100][:5]),
            "last_five_evaluations_mean": mean(mean(r["returns"]) for r in curves[100][-5:]),
        },
        "floor_mean": floor_mean,
        "floor_episode_sd": pstdev(floor["returns"]),
        "final_five_episode_means_across_training_seeds": {
            "mean": mean(final_returns), "sample_sd": stdev(final_returns),
        },
        "runs": {},
    }
    table = [
        "| Task / policy | Training seed | Final 20-episode return | Training control steps | GPU-hours | Peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, manifest in manifests.items():
        final = curves[seed][-1]
        run = {
            "final_5_episode_mean": mean(final["returns"]),
            "final_5_episode_sd": pstdev(final["returns"]),
            "last_five_evaluations_mean": mean(mean(r["returns"]) for r in curves[seed][-5:]),
            "training_control_steps": manifest["counters"]["env_step"],
            "wall_hours": manifest["wall_seconds"] / 3600,
            "peak_vram_mib": manifest["peak_vram_mib"],
            "trainable_parameters": manifest["parameters"]["trainable_total"],
        }
        if seed == 100:
            returns = controls["returns"]["trained"]
            run["final_20_episode_mean"] = mean(returns)
            run["final_20_episode_sd"] = pstdev(returns)
            score = f"{mean(returns):.2f} ± {pstdev(returns):.2f}"
        else:
            score = "Not measured"
        stats["runs"][str(seed)] = run
        hours = f"{run['wall_hours']:.3f}" + ("†" if seed == 100 else "")
        table.append(f"| Walker Walk, H=15 | {seed} | {score} | {run['training_control_steps']:,} | {hours} | {run['peak_vram_mib']:,.1f} MiB |")
    random = controls["returns"]["random"]
    table.append(f"| Walker Walk, uniform random | — | {mean(random):.2f} ± {pstdev(random):.2f} | 0 (no training) | — | — |")
    stats["matched_random"] = {"mean": mean(random), "episode_sd": pstdev(random)}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172033",
        "axes.edgecolor": "#cbd5e1", "xtick.color": "#475569", "ytick.color": "#475569",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })
    colors = {100: "#2563eb", 101: "#c25a16", 102: "#0d8b70"}
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.17)
    fig.text(.09, .94, "Walker Walk learns from pixels", fontsize=18, weight="bold")
    fig.text(.09, .885, "H=15 · 686,846 trainable parameters · three development training seeds", fontsize=11, color="#475569")
    for seed, rows in curves.items():
        ax.plot([r["env_step"] for r in rows], [mean(r["returns"]) for r in rows],
                color=colors[seed], lw=2, label=f"Seed {seed}")
    ax.axhline(floor_mean, color="#64748b", lw=1.4, ls="--", label=f"Random floor ({floor_mean:.2f})")
    ax.set(xlim=(0, 1_000_000), ylim=(0, 650), xlabel="Training control steps", ylabel="Episode return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "1M" if x == 1_000_000 else f"{x/1000:.0f}K" if x else "0"))
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.text(.09, .035, "Each point: mean of 5 episodes at fixed evaluation seeds 2000–2004. Raw curves; no smoothing.", fontsize=10, color="#475569")
    for extension in ("png", "svg"):
        fig.savefig(out / f"walker-learning-curve.{extension}", dpi=180)
    plt.close(fig)

    diag = controls["openloop"]
    distance = range(1, len(diag["reward_mae"]) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.19)
    fig.text(.09, .94, "Action-conditioned reward prediction", fontsize=18, weight="bold")
    fig.text(.09, .885, "Walker seed 100 final checkpoint · on-policy trajectories · prior-only future rollout", fontsize=11, color="#475569")
    for key, label, color, style in (
        ("reward_mae", "Model, recorded actions", "#2563eb", "-"),
        ("reward_mae_shuffled", "Model, shuffled actions", "#c25a16", "-"),
        ("baseline_reward_mae_persistence", "Last-reward persistence", "#0d8b70", "--"),
        ("baseline_reward_mae_mean", "Constant reward (in-sample)", "#64748b", ":"),
    ):
        ax.plot(distance, diag[key], label=label, color=color, ls=style, lw=2)
    ax.set(xlim=(1, 30), ylim=(0, .55), xlabel="Prediction distance (control steps)", ylabel="Reward mean absolute error")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
    fig.text(.09, .065, f"{diag['contexts']} contexts · {diag['context_length']}-transition context · {diag['samples']} latent samples. Same trajectories for every curve.", fontsize=10, color="#475569")
    fig.text(.09, .025, "Constant baseline uses the evaluated trajectories; this is not the reserved final diagnostic corpus.", fontsize=10, color="#475569")
    for extension in ("png", "svg"):
        fig.savefig(out / f"walker-prediction-error.{extension}", dpi=180)
    plt.close(fig)
    stats["prediction"] = {
        "context_length": diag["context_length"], "contexts": diag["contexts"], "samples": diag["samples"],
        "horizon": diag["horizon"], "reward_mae_k30": diag["reward_mae"][-1],
        "shuffled_reward_mae_k30": diag["reward_mae_shuffled"][-1],
        "shuffled_error_ratio_k30": diag["reward_mae_shuffled"][-1] / diag["reward_mae"][-1],
    }
    (out / "readme-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "readme-results-table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"Wrote figures, table, and source hashes to {out}")


if __name__ == "__main__":
    main()
