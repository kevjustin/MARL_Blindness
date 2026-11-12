# MARL Blindness

How does temporary observation loss affect coordination in multi-agent reinforcement learning, and does telling an agent when its observations are masked help? This project explores those questions in the VMAS `balance` scenario using a PyTorch implementation of Multi-Agent Proximal Policy Optimization (MAPPO). The agents share a policy that acts on individual observations, while a centralized critic uses the agents' combined observations during training.

## Experiment setup

When no blindness event is active, each eligible environment step has probability `p` of starting one. The simulator selects one agent at random and replaces its observations for up to `k` steps. Masked observations are replaced with zeros by default. Gaussian noise is also available.

| Configuration | Start probability | Duration | Blindness indicator |
| --- | ---: | ---: | --- |
| Baseline | 0% | 0 steps | Disabled |
| Naive blind | 5% | 10 steps | Disabled |
| Self-aware blind | 5% | 10 steps | Appended to each agent's observation |

## Installation

From the repository root, install the dependencies and Matplotlib, which the plotting scripts import.

```bash
python -m pip install -r requirements.txt matplotlib
```

## Run the comparison

```bash
python run_experiments.py
```

The script trains all three configurations on CPU with a budget of 300,000 environment steps per run. It displays learning curves and prints the final mean episodic returns. The curves show the cumulative average over completed training episodes. The final values use all completed training episodes.

## Run a custom experiment

```bash
python blindness_mappo_vmas.py --scenario balance --device cpu --total_steps 100000 --blind_prob 0.05 --blind_k 10 --blind_self_flag
```

Omit `--blind_self_flag` to run the naive condition, or set `--blind_prob 0` to run the baseline. The training CLI also accepts `--scenario wheel` and `--blind_mask noise`.

## Plot parameter sweeps

`plot_results.py` plots final mean episodic return against blindness probability at `k = 10` and against event duration at `p = 0.10`. Supply a `results_balance.csv` in the repository root with the header below. The comparison script displays learning curves and prints returns. Sweep data must be collected separately.

```csv
algo,blind_prob,blind_k,blind_self_flag,final_return
```

Use `baseline`, `naive`, or `selfaware` for `algo` and `0` or `1` for `blind_self_flag`. Then run:

```bash
python plot_results.py
```

## Repository files

| File | Purpose |
| --- | --- |
| `blindness_mappo_vmas.py` | Defines the blindness wrapper, shared policy, centralized critic, and PPO training loop. |
| `run_experiments.py` | Runs the three default configurations and displays their learning curves. |
| `plot_results.py` | Reads a supplied CSV and displays the parameter sweep plots. |
| `requirements.txt` | Lists the core Python dependencies. |

## Implementation note

The current wrapper clears the event state before appending the indicator on the final masked step. That step therefore has a masked observation and an indicator of `0`.
