"""
training/train_ppo.py
──────────────────────
Phase 3 — PPO training loop.

Key structural difference from train_reinforce.py:
  REINFORCE: for each episode → collect → update
  PPO:       for each step   → collect → IF buffer full → update

The buffer drives the update cadence, not episode boundaries.
This is cleaner and handles multi-episode rollouts naturally.

Usage:
    python -m training.train_ppo --skus 10 --steps 10000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from envs.retail_env import RetailEnv
from agents.ppo_agent import PPOAgent
from config.settings import TrainingConfig

console = Console()


def train(
    n_skus: int = 20,
    total_steps: int = 40_960,   # 20 updates × 2048 steps each
    n_steps_per_rollout: int = 2048,
    seed: int = 42,
    save_dir: str = "checkpoints",
) -> PPOAgent:
    """
    Train PPO agent on QuickMart environment.

    Args:
        n_skus              : SKUs to use (subset of 425 for speed).
        total_steps         : Total env steps to train for.
        n_steps_per_rollout : Steps collected before each PPO update.
        seed                : Random seed.
        save_dir            : Checkpoint directory.

    Returns:
        Trained PPOAgent.
    """
    Path(save_dir).mkdir(exist_ok=True)
    cfg = TrainingConfig(seed=seed)

    # ── Environment
    env = RetailEnv(n_skus=n_skus, max_steps=365, seed=seed)
    obs, _ = env.reset(seed=seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── Agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=cfg,
        n_steps=n_steps_per_rollout,
        device=cfg.device,
    )

    console.print(f"\n[bold cyan]Phase 3: PPO Training[/bold cyan]")
    console.print(f"SKUs: {n_skus}  |  obs_dim: {obs_dim}  |  action_dim: {action_dim}")
    console.print(f"Rollout steps: {n_steps_per_rollout}  |  Total steps: {total_steps}")
    console.print(f"Epochs/update: {cfg.n_epochs_per_update}  |  Minibatch: {cfg.minibatch_size}\n")

    episode_rewards: list = []
    ep_reward = 0.0
    update_metrics = []

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        task = progress.add_task("[green]Training...", total=total_steps)

        for step in range(total_steps):

            # ── Act
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ── Store (pass done so GAE handles episode boundaries)
            agent.store_transition(action, reward, done=done)
            ep_reward += reward

            # ── Handle episode end
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

            # ── Update when buffer is full
            if agent.buffer_full():
                # Pass last obs for value bootstrapping
                last_obs = obs
                metrics = agent.update(last_obs=last_obs)
                update_metrics.append(metrics)

                if len(update_metrics) % 5 == 0:
                    recent_eps = episode_rewards[-10:] if episode_rewards else [0]
                    progress.stop()
                    console.print(
                        f"  Update {metrics['update']:>4} | "
                        f"Steps: {metrics['total_steps']:>8,} | "
                        f"π loss: {metrics['policy_loss']:>8.4f} | "
                        f"V loss: {metrics['value_loss']:>8.4f} | "
                        f"Entropy: {metrics['entropy']:>6.4f} | "
                        f"Avg reward: {np.mean(recent_eps):>10.1f}"
                    )
                    progress.start()

            progress.advance(task)

    # ── Summary
    console.print("\n[bold green]PPO Training complete![/bold green]")
    _print_summary(episode_rewards, update_metrics)

    agent.save(os.path.join(save_dir, "ppo_final.pt"))
    return agent


def _print_summary(episode_rewards: list, update_metrics: list) -> None:
    if not episode_rewards:
        console.print("[yellow]No complete episodes recorded.[/yellow]")
        return

    first_10 = np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    last_10  = np.mean(episode_rewards[-10:])
    improvement = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0.0

    table = Table(title="PPO Training Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Total episodes", str(len(episode_rewards)))
    table.add_row("Total updates", str(len(update_metrics)))
    table.add_row("Mean reward (first 10 eps)", f"{first_10:.1f}")
    table.add_row("Mean reward (last  10 eps)", f"{last_10:.1f}")
    table.add_row("Improvement", f"{improvement:+.1f}%")
    table.add_row("Best episode reward", f"{max(episode_rewards):.1f}")

    if update_metrics:
        last_m = update_metrics[-1]
        table.add_row("Final policy loss", f"{last_m['policy_loss']:.4f}")
        table.add_row("Final value loss",  f"{last_m['value_loss']:.4f}")
        table.add_row("Final entropy",     f"{last_m['entropy']:.4f}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--skus",   type=int, default=20)
    parser.add_argument("--steps",  type=int, default=40_960)
    parser.add_argument("--rollout",type=int, default=2048)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    train(
        n_skus=args.skus,
        total_steps=args.steps,
        n_steps_per_rollout=args.rollout,
        seed=args.seed,
    )