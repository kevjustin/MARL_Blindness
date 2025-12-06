'''
plot_results.py

Reads results_balance.csv and produces:
  Figure 1: final_return vs blind_prob (p-sweep at k=10)
  Figure 2: final_return vs blind_k (k-sweep at p=0.10)
Make sure run_experiments.py has run first
'''

import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "results_balance.csv"


def load_results(csv_path):
    """
    Load CSV into a list of dicts, converting numeric fields.
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["blind_prob"] = float(row["blind_prob"])
            row["blind_k"] = int(row["blind_k"])
            row["blind_self_flag"] = int(row["blind_self_flag"])
            row["final_return"] = float(row["final_return"])
            rows.append(row)
    return rows


def plot_p_sweep_k10(results):
    """
    Plot final_return vs blind_prob for runs with blind_k = 10.

    - Baseline: plotted as a horizontal reference line.
    - Naive and self-aware: line plots over p.
    """
    # Extract baseline (blind_prob = 0.0, algo="baseline")
    baseline_values = [
        r["final_return"]
        for r in results
        if r["algo"] == "baseline" and r["blind_prob"] == 0.0
    ]
    baseline_return = baseline_values[0] if baseline_values else None

    # Filter k=10 for naive and selfaware
    naive_rows = [
        r
        for r in results
        if r["algo"] == "naive" and r["blind_k"] == 10 and r["blind_prob"] > 0.0
    ]
    self_rows = [
        r
        for r in results
        if r["algo"] == "selfaware" and r["blind_k"] == 10 and r["blind_prob"] > 0.0
    ]

    # Sort by blind_prob
    naive_rows = sorted(naive_rows, key=lambda r: r["blind_prob"])
    self_rows = sorted(self_rows, key=lambda r: r["blind_prob"])

    p_naive = [r["blind_prob"] for r in naive_rows]
    y_naive = [r["final_return"] for r in naive_rows]

    p_self = [r["blind_prob"] for r in self_rows]
    y_self = [r["final_return"] for r in self_rows]

    if len(p_naive) == 0 or len(p_self) == 0:
        print("No naive/selfaware runs with k=10 found; skipping p-sweep plot.")
        return

    plt.figure()
    # Naive and self-aware lines
    plt.plot(p_naive, y_naive, marker="o", label="Naive blind (k=10)")
    plt.plot(p_self, y_self, marker="o", label="Self-aware (k=10)")

    # Baseline as horizontal line across the p-range
    if baseline_return is not None:
        p_min = min(p_naive + p_self)
        p_max = max(p_naive + p_self)
        plt.hlines(
            baseline_return,
            p_min,
            p_max,
            linestyles="dashed",
            label=f"Baseline (p=0) = {baseline_return:.2f}",
        )

    plt.xlabel("Blind probability p")
    plt.ylabel("Final mean episodic return")
    plt.title("Balance scenario: performance vs blindness probability (k=10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_k_sweep_pmid(results, p_mid=0.10):
    """
    Plot final_return vs blind_k for a fixed blind_prob = p_mid.

    Shows naive vs self-aware.
    """
    naive_rows = [
        r for r in results if r["algo"] == "naive" and abs(r["blind_prob"] - p_mid) < 1e-8
    ]
    self_rows = [
        r
        for r in results
        if r["algo"] == "selfaware" and abs(r["blind_prob"] - p_mid) < 1e-8
    ]

    naive_rows = sorted(naive_rows, key=lambda r: r["blind_k"])
    self_rows = sorted(self_rows, key=lambda r: r["blind_k"])

    k_naive = [r["blind_k"] for r in naive_rows]
    y_naive = [r["final_return"] for r in naive_rows]

    k_self = [r["blind_k"] for r in self_rows]
    y_self = [r["final_return"] for r in self_rows]

    if len(k_naive) == 0 or len(k_self) == 0:
        print(f"No naive/selfaware runs at p={p_mid} found; skipping k-sweep plot.")
        return

    ks = sorted(set(k_naive + k_self))

    # We assume we have the same ks for both; if not, we could align them more carefully.
    x = np.arange(len(ks))
    width = 0.35

    # Build aligned y arrays
    naive_by_k = {k: v for k, v in zip(k_naive, y_naive)}
    self_by_k = {k: v for k, v in zip(k_self, y_self)}
    naive_vals = [naive_by_k.get(k, np.nan) for k in ks]
    self_vals = [self_by_k.get(k, np.nan) for k in ks]

    plt.figure()
    plt.bar(x - width / 2, naive_vals, width, label="Naive blind")
    plt.bar(x + width / 2, self_vals, width, label="Self-aware")

    plt.xticks(x, [str(k) for k in ks])
    plt.xlabel("Blind duration k (steps)")
    plt.ylabel("Final mean episodic return")
    plt.title(f"Balance scenario: performance vs blind duration (p={p_mid})")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()


def main():
    results = load_results(CSV_PATH)
    print(f"Loaded {len(results)} rows from {CSV_PATH}")

    plot_p_sweep_k10(results)
    plot_k_sweep_pmid(results, p_mid=0.10)

    plt.show()


if __name__ == "__main__":
    main()
