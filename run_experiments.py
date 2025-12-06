'''
run_experiments.py

Tester File to run three different configurations of the blindness-aware
multi-agent PPO (MAPPO) algorithm on the VMAS "balance" scenario:
  1) Baseline
  2) Naive blind
  3) Self-aware blind
and plots their learning curves on one graph.
'''

from blindness_mappo_vmas import Config, train
import matplotlib.pyplot as plt

def run_one(name: str, cfg: Config):
    print("=" * 80)
    print(f"Running: {name}")
    final_return, curve = train(cfg)
    print(f"{name} final return: {final_return:.3f}")
    return final_return, curve

def main():
    scenario = "balance"
    blind_k = 10
    total_steps = 300_000

    # Baseline (no blindness)
    cfg_base = Config(
        scenario=scenario,
        device="cpu",
        total_steps=total_steps,
        blind_prob=0.0,
        blind_k=0,
        blind_self_flag=False,
    )
    base_final, base_curve = run_one("BASELINE (p=0.0)", cfg_base)

    # Naive blind
    cfg_naive = Config(
        scenario=scenario,
        device="cpu",
        total_steps=total_steps,
        blind_prob=0.05,
        blind_k=blind_k,
        blind_self_flag=False,
    )
    naive_final, naive_curve = run_one("NAIVE BLIND (p=0.05, k=10)", cfg_naive)

    # Self-aware blind
    cfg_self = Config(
        scenario=scenario,
        device="cpu",
        total_steps=total_steps,
        blind_prob=0.05,
        blind_k=blind_k,
        blind_self_flag=True,
    )
    self_final, self_curve = run_one("SELF-AWARE BLIND (p=0.05, k=10)", cfg_self)

    # Plot learning curves
    plt.figure(figsize=(10, 6))

    # Each curve is a list of (step, overall_mean)
    if base_curve:
        base_steps, base_vals = zip(*base_curve)
        plt.plot(base_steps, base_vals, label=f"Baseline (p=0.0), final={base_final:.2f}")
    if naive_curve:
        naive_steps, naive_vals = zip(*naive_curve)
        plt.plot(naive_steps, naive_vals, label=f"Naive blind (p=0.05), final={naive_final:.2f}")
    if self_curve:
        self_steps, self_vals = zip(*self_curve)
        plt.plot(self_steps, self_vals, label=f"Self-aware (p=0.05), final={self_final:.2f}")

    plt.xlabel("Training steps")
    plt.ylabel("Overall mean episodic return")
    plt.title(f"Learning curves (scenario={scenario}, k={blind_k})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Prints final numbers
    print("\nFinal mean episodic returns:")
    print(f"Baseline (p=0.0):         {base_final:.3f}")
    print(f"Naive blind (p=0.05):     {naive_final:.3f}")
    print(f"Self-aware (p=0.05):      {self_final:.3f}")

if __name__ == "__main__":
    main()
