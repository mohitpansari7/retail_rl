"""
main.py
────────
Entry point for the RetailRL project.

Usage:
    python main.py                  # runs Phase 1 smoke test
    python main.py --phase 2        # (future phases)

Currently wired to Phase 1: spins up the environment, runs one episode
with a random policy, and prints a summary.  Each phase adds its own
runner here.
"""

import argparse
import numpy as np
from rich.console import Console
from rich.table import Table

from envs.retail_env import RetailEnv
from config.settings import TrainingConfig

console = Console()


def run_random_baseline(n_skus: int = 20, max_steps: int = 30) -> None:
    """
    Phase 1 smoke test: run one episode with a random policy.
    Validates that the MDP is wired correctly before any learning.
    """
    cfg = TrainingConfig()
    env = RetailEnv(n_skus=n_skus, max_steps=max_steps, seed=cfg.seed, render_mode="human")

    obs, info = env.reset(seed=cfg.seed)
    console.print(f"\n[bold cyan]RetailRL — Phase 1 Smoke Test[/bold cyan]")
    console.print(f"Store : {env.store_id}")
    console.print(f"SKUs  : {env.n_skus}  |  Steps: {max_steps}")
    console.print(f"Obs shape: {obs.shape}\n")

    total_reward = 0.0
    episode_revenues = []

    for step in range(max_steps):
        action = env.action_space.sample()          # random price changes
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_revenues.append(info["total_revenue"])
        env.render()
        if terminated or truncated:
            break

    # ── Summary table
    table = Table(title="Episode Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Total steps", str(step + 1))
    table.add_row("Cumulative reward", f"{total_reward:,.2f}")
    table.add_row("Total revenue (₹)", f"{info['episode_revenue']:,.2f}")
    table.add_row("Avg revenue/step (₹)", f"{np.mean(episode_revenues):,.2f}")
    table.add_row("Stockouts (last step)", str(info["stockouts"]))
    console.print(table)

    console.print("\n[bold green]✓ Phase 1 environment verified.[/bold green]")
    console.print("Next: implement Phase 2 (REINFORCE agent).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailRL runner")
    parser.add_argument("--phase", type=int, default=1, help="Phase to run (1–10)")
    parser.add_argument("--skus", type=int, default=20, help="Number of SKUs (subset for speed)")
    parser.add_argument("--steps", type=int, default=30, help="Steps per episode")
    args = parser.parse_args()

    if args.phase == 1:
        run_random_baseline(n_skus=args.skus, max_steps=args.steps)
    else:
        console.print(f"[yellow]Phase {args.phase} not yet implemented. Coming soon![/yellow]")