# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION &
# AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Study MLE performance for cardinality estimation under uniform sampling."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from statistics import mean, pstdev

import numpy as np

try:  # Optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None

try:  # Optional dependency
    from scipy import stats
except Exception:  # pragma: no cover - optional scipy
    stats = None


@dataclass(frozen=True)
class TrialResult:
    true_m: int
    n: int
    est_m: float
    distinct_seen: int


def make_weights(
    true_m: int,
    dist: str,
    zipf_s: float,
    head_fraction: float,
    head_mass: float,
    lognormal_mu: float,
    lognormal_sigma: float,
    gamma_shape: float,
    gamma_scale: float,
    pareto_b: float,
    pareto_scale: float,
    weibull_c: float,
    weibull_scale: float,
    weight_seed: int,
) -> list[float] | None:
    if dist == "uniform":
        return None
    random_rng = random.Random(weight_seed)
    numpy_rng = np.random.default_rng(weight_seed)
    if dist == "zipf":
        weights = [1.0 / ((idx + 1) ** zipf_s) for idx in range(true_m)]
        total = sum(weights)
        return [weight / total for weight in weights]
    if dist == "head":
        head = max(1, int(true_m * head_fraction))
        tail = true_m - head
        weights: list[float] = []
        head_weight = head_mass / head
        tail_weight = (1.0 - head_mass) / tail if tail > 0 else 0.0
        weights.extend([head_weight] * head)
        weights.extend([tail_weight] * tail)
        return weights
    if dist == "lognormal":
        weights = [
            random_rng.lognormvariate(lognormal_mu, lognormal_sigma)
            for _ in range(true_m)
        ]
        total = sum(weights)
        return [weight / total for weight in weights]
    if dist == "gamma":
        if stats is None:
            raise RuntimeError("scipy is required for gamma distribution")
        weights = stats.gamma.rvs(
            gamma_shape, scale=gamma_scale, size=true_m, random_state=numpy_rng
        )
        total = float(np.sum(weights))
        return (weights / total).tolist()
    if dist == "pareto":
        if stats is None:
            raise RuntimeError("scipy is required for pareto distribution")
        weights = stats.pareto.rvs(
            pareto_b, scale=pareto_scale, size=true_m, random_state=numpy_rng
        )
        total = float(np.sum(weights))
        return (weights / total).tolist()
    if dist == "weibull":
        if stats is None:
            raise RuntimeError("scipy is required for weibull distribution")
        weights = stats.weibull_min.rvs(
            weibull_c, scale=weibull_scale, size=true_m, random_state=numpy_rng
        )
        total = float(np.sum(weights))
        return (weights / total).tolist()
    raise ValueError(f"Unsupported dist: {dist}")


def make_weight_seed(
    base_seed: int,
    true_m: int,
    dist: str,
    zipf_s: float,
    head_fraction: float,
    head_mass: float,
    lognormal_mu: float,
    lognormal_sigma: float,
    gamma_shape: float,
    gamma_scale: float,
    pareto_b: float,
    pareto_scale: float,
    weibull_c: float,
    weibull_scale: float,
) -> int:
    seed = hash(
        (
            base_seed,
            true_m,
            dist,
            zipf_s,
            head_fraction,
            head_mass,
            lognormal_mu,
            lognormal_sigma,
            gamma_shape,
            gamma_scale,
            pareto_b,
            pareto_scale,
            weibull_c,
            weibull_scale,
        )
    )
    return seed & 0xFFFFFFFF


def simulate_distinct_count(
    true_m: int, n: int, rng: random.Random, weights: list[float] | None
) -> int:
    if weights is None:
        draws = (rng.randrange(true_m) for _ in range(n))
    else:
        draws = rng.choices(range(true_m), weights=weights, k=n)
    return len(set(draws))


def simulate_counts(
    true_m: int, n: int, rng: random.Random, weights: list[float] | None
) -> dict[int, int]:
    if weights is None:
        draws = (rng.randrange(true_m) for _ in range(n))
    else:
        draws = rng.choices(range(true_m), weights=weights, k=n)
    counts: dict[int, int] = {}
    for item in draws:
        counts[item] = counts.get(item, 0) + 1
    return counts


def mle_estimate_from_distinct(distinct_seen: int, n: int) -> float:
    if distinct_seen <= 0:
        return 0.0
    if distinct_seen >= n:
        return float("inf")

    def f(m: float) -> float:
        return m * (1.0 - (1.0 - 1.0 / m) ** n) - distinct_seen

    low = float(distinct_seen)
    high = max(low * 2.0, n * 2.0)
    while f(high) < 0.0 and high < 1e12:
        high *= 2.0

    if not math.isfinite(high):
        return float("inf")

    for _ in range(80):
        mid = 0.5 * (low + high)
        if f(mid) >= 0.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def run_trials(
    true_m: int, n: int, trials: int, rng: random.Random, weights: list[float] | None
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for _ in range(trials):
        distinct_seen = simulate_distinct_count(true_m, n, rng, weights)
        est_m = mle_estimate_from_distinct(distinct_seen, n)
        results.append(
            TrialResult(true_m=true_m, n=n, est_m=est_m, distinct_seen=distinct_seen)
        )
    return results


def summarize(results: list[TrialResult]) -> tuple[float, float, float, float, float]:
    finite = [r for r in results if math.isfinite(r.est_m)]
    if not finite:
        return (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
    estimates = [r.est_m for r in finite]
    rel_errors = [(r.est_m - r.true_m) / r.true_m for r in finite]
    abs_rel_errors = [abs(err) for err in rel_errors]
    return (
        mean(estimates),
        pstdev(estimates),
        mean(rel_errors),
        pstdev(rel_errors),
        mean(abs_rel_errors),
    )


def plot_summary(
    summaries: list[tuple[int, int, float, float, float, float]],
    dist: str,
    output: str | None,
    dist_weights: list[float] | None,
    reference_m: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    summaries.sort(key=lambda item: (item[0], item[1]))
    ms = sorted({item[0] for item in summaries})
    ns = sorted({item[1] for item in summaries})

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))

    for true_m in ms:
        mean_values = [item[2] for item in summaries if item[0] == true_m]
        std_values = [item[3] for item in summaries if item[0] == true_m]
        distinct_values = [item[4] for item in summaries if item[0] == true_m]
        abs_rel_values = [item[5] for item in summaries if item[0] == true_m]
        axes[0, 0].plot(ns, mean_values, marker="o", label=f"M={true_m}")
        axes[0, 1].plot(ns, std_values, marker="o", label=f"M={true_m}")
        axes[1, 0].plot(ns, distinct_values, marker="o", label=f"M={true_m}")
        axes[1, 1].plot(ns, abs_rel_values, marker="o", label=f"M={true_m}")

    for axis, label in (
        (axes[0, 0], "Mean signed relative error"),
        (axes[0, 1], "Std signed relative error"),
        (axes[1, 0], "Mean distinct observed"),
        (axes[1, 1], "Mean absolute relative error"),
    ):
        axis.set_xscale("log")
        axis.set_xlabel("n (log scale)")
        axis.set_ylabel(label)

    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title(f"MLE bias ({dist})")
    axes[0, 1].set_title(f"MLE variability ({dist})")
    axes[1, 0].set_title(f"Coverage ({dist})")
    axes[1, 1].set_title(f"Abs relative error ({dist})")
    axes[0, 0].legend()

    axes[0, 2].set_title(f"Draw distribution ({dist})")
    if dist_weights is None:
        max_rank = min(reference_m, 2000)
        ranks = list(range(1, max_rank + 1))
        axes[0, 2].plot(ranks, [1.0 / reference_m] * max_rank)
        axes[0, 2].set_xscale("log")
    else:
        sorted_weights = sorted(dist_weights, reverse=True)
        max_points = 2000
        if len(sorted_weights) > max_points:
            stride = len(sorted_weights) // max_points
            sorted_weights = sorted_weights[::stride]
        ranks = list(range(1, len(sorted_weights) + 1))
        axes[0, 2].plot(ranks, sorted_weights)
        axes[0, 2].set_xscale("log")
        axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("Rank")
    axes[0, 2].set_ylabel("Probability")
    axes[1, 2].axis("off")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


def plot_heatmap(
    summaries: list[tuple[int, int, float, float, float, float]],
    dist: str,
    output: str | None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    ms = sorted({item[0] for item in summaries})
    ns = sorted({item[1] for item in summaries})
    grid = [[0.0 for _ in ns] for _ in ms]
    for m_idx, true_m in enumerate(ms):
        for n_idx, n in enumerate(ns):
            for item in summaries:
                if item[0] == true_m and item[1] == n:
                    grid[m_idx][n_idx] = item[2]
                    break

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xticks(range(len(ns)), labels=[str(n) for n in ns])
    ax.set_yticks(range(len(ms)), labels=[str(m) for m in ms])
    ax.set_xlabel("n")
    ax.set_ylabel("M")
    ax.set_title(f"Mean signed relative error heatmap ({dist})")
    fig.colorbar(image, ax=ax, label="Mean signed relative error")
    fig.tight_layout()

    if output:
        stem, suffix = (output.rsplit(".", 1) + ["png"])[:2]
        fig.savefig(f"{stem}_heatmap.{suffix}", dpi=150)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the MLE for cardinality under iid uniform sampling. "
            "Reports mean/std of estimates and relative error."
        )
    )
    parser.add_argument("--ms", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--ns", nargs="+", type=int, default=[500, 1000, 5000])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dist",
        choices=("uniform", "zipf", "head", "lognormal", "gamma", "pareto", "weibull"),
        default="uniform",
        help="Distribution for draws over M distinct values.",
    )
    parser.add_argument(
        "--zipf-s",
        type=float,
        default=1.1,
        help="Zipf exponent (larger means more skew).",
    )
    parser.add_argument(
        "--head-fraction",
        type=float,
        default=0.1,
        help="Fraction of items in head for the head distribution.",
    )
    parser.add_argument(
        "--head-mass",
        type=float,
        default=0.8,
        help="Probability mass assigned to the head for the head distribution.",
    )
    parser.add_argument(
        "--lognormal-mu",
        type=float,
        default=0.0,
        help="Lognormal mu parameter for the lognormal distribution.",
    )
    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=1.0,
        help="Lognormal sigma parameter for the lognormal distribution.",
    )
    parser.add_argument(
        "--gamma-shape",
        type=float,
        default=1.5,
        help="Gamma shape (k) parameter.",
    )
    parser.add_argument(
        "--gamma-scale",
        type=float,
        default=1.0,
        help="Gamma scale (theta) parameter.",
    )
    parser.add_argument(
        "--pareto-b",
        type=float,
        default=2.5,
        help="Pareto shape parameter.",
    )
    parser.add_argument(
        "--pareto-scale",
        type=float,
        default=1.0,
        help="Pareto scale parameter.",
    )
    parser.add_argument(
        "--weibull-c",
        type=float,
        default=1.5,
        help="Weibull shape parameter.",
    )
    parser.add_argument(
        "--weibull-scale",
        type=float,
        default=1.0,
        help="Weibull scale parameter.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot summary panels vs n (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Write plot to this file instead of showing it.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Add a heatmap for mean relative error over (M, n).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    weights_by_m: dict[int, list[float] | None] = {}

    header = (
        "dist,M,n,trials,mean_estimate,std_estimate,mean_rel_error,std_rel_error,"
        "mean_abs_rel_error,mean_distinct_seen"
    )
    print(header)
    summaries: list[tuple[int, int, float, float, float, float]] = []
    for true_m in args.ms:
        if true_m not in weights_by_m:
            weights_seed = make_weight_seed(
                args.seed,
                true_m,
                args.dist,
                args.zipf_s,
                args.head_fraction,
                args.head_mass,
                args.lognormal_mu,
                args.lognormal_sigma,
                args.gamma_shape,
                args.gamma_scale,
                args.pareto_b,
                args.pareto_scale,
                args.weibull_c,
                args.weibull_scale,
            )
            weights_by_m[true_m] = make_weights(
                true_m,
                args.dist,
                args.zipf_s,
                args.head_fraction,
                args.head_mass,
                args.lognormal_mu,
                args.lognormal_sigma,
                args.gamma_shape,
                args.gamma_scale,
                args.pareto_b,
                args.pareto_scale,
                args.weibull_c,
                args.weibull_scale,
                weights_seed,
            )
        weights = weights_by_m[true_m]
        for n in args.ns:
            results = run_trials(true_m, n, args.trials, rng, weights)
            mean_est, std_est, mean_rel, std_rel, mean_abs_rel = summarize(results)
            mean_distinct = mean(r.distinct_seen for r in results)
            print(
                f"{args.dist},{true_m},{n},{args.trials},"
                f"{mean_est:.6g},{std_est:.6g},"
                f"{mean_rel:.6g},{std_rel:.6g},{mean_abs_rel:.6g},"
                f"{mean_distinct:.6g}"
            )
            summaries.append(
                (true_m, n, mean_rel, std_rel, mean_distinct, mean_abs_rel)
            )

    if args.plot:
        reference_m = args.ms[0] if args.ms else 1
        dist_weights = weights_by_m.get(reference_m)
        plot_summary(
            summaries,
            args.dist,
            args.plot_output,
            dist_weights,
            reference_m,
        )

    if args.heatmap:
        plot_heatmap(summaries, args.dist, args.plot_output)


if __name__ == "__main__":
    main()
