"""
training/train_reinforce.py
────────────────────────────
Phase 2 — Training loop for REINFORCE agent.

This is the outer loop that wires together:
    env (RetailEnv) ↔ agent (ReinforceAgent)

Training loop structure:
    for episode in range(N):
        obs = env.reset()
        for step in range(max_steps):
            action = agent.select_action(obs)
            obs, reward, done, _, info = env.step(action)
            agent.store_transition(action, reward)
        metrics = agent.update()   ← update AFTER full episode
        log metrics

Usage:
    python -m training.train_reinforce
    python -m training.train_reinforce --skus 10 --episodes 50
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
from agents.reinforce_agent import ReinforceAgent
from config.settings import TrainingConfig

console = Console()


def train(
    n_skus: int = 20,
    n_episodes: int = 200,
    max_steps: int = 60,
    seed: int = 42,
    save_dir: str = "checkpoints",
) -> ReinforceAgent:
    """
    Train a REINFORCE agent on the QuickMart environment.

    Args:
        n_skus     : Number of SKUs (subset for speed; full = 425)
        n_episodes : Total training episodes
        max_steps  : Steps per episode (days of operation)
        seed       : Random seed for reproducibility
        save_dir   : Where to save checkpoints

    Returns:
        Trained ReinforceAgent.
    """
    Path(save_dir).mkdir(exist_ok=True)
    cfg = TrainingConfig(seed=seed, n_episodes=n_episodes, max_steps_per_episode=max_steps)

    # ── Environment
    env = RetailEnv(n_skus=n_skus, max_steps=max_steps, seed=seed)
    obs, _ = env.reset(seed=seed)

    # ── Agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ReinforceAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=cfg,
        device=cfg.device,
    )

    console.print(f"\n[bold cyan]Phase 2: REINFORCE Training[/bold cyan]")
    console.print(f"SKUs: {n_skus}  |  obs_dim: {obs_dim}  |  action_dim: {action_dim}")
    console.print(f"Episodes: {n_episodes}  |  Steps/episode: {max_steps}\n")

    # ── Training loop
    episode_returns = []
    episode_lengths = []

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        task = progress.add_task("[green]Training...", total=n_episodes)

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_steps = 0

            # ── Collect one full episode
            for step in range(max_steps):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.store_transition(action, float(np.clip(reward / 1e7, -10.0, 10.0)))
                ep_reward += reward
                ep_steps += 1
                if terminated or truncated:
                    break

            # ── Update policy AFTER full episode (Monte Carlo)
            metrics = agent.update()

            episode_returns.append(ep_reward)
            episode_lengths.append(ep_steps)
            progress.advance(task)

            # ── Periodic logging
            if (ep + 1) % cfg.log_interval == 0:
                window = 10
                recent_returns = episode_returns[-window:]
                progress.stop()
                console.print(
                    f"  Ep {ep+1:>4} | "
                    f"Return: {ep_reward:>10.1f} | "
                    f"Avg({window}): {np.mean(recent_returns):>10.1f} | "
                    f"π loss: {metrics.get('policy_loss', 0):>7.4f} | "
                    f"V loss: {metrics.get('value_loss', 0):>7.4f}"
                )
                progress.start()

            # ── Save checkpoint
            if (ep + 1) % cfg.checkpoint_interval == 0:
                ckpt_path = os.path.join(save_dir, f"reinforce_ep{ep+1}.pt")
                agent.save(ckpt_path)

    # ── Final summary
    console.print("\n[bold green]Training complete![/bold green]")
    _print_summary(episode_returns, n_skus, n_episodes, max_steps)

    agent.save(os.path.join(save_dir, "reinforce_final.pt"))
    return agent


def _print_summary(
    episode_returns: list,
    n_skus: int,
    n_episodes: int,
    max_steps: int,
) -> None:
    first_10 = np.mean(episode_returns[:10])
    last_10 = np.mean(episode_returns[-10:])
    improvement = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0

    table = Table(title="REINFORCE Training Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("SKUs", str(n_skus))
    table.add_row("Episodes", str(n_episodes))
    table.add_row("Steps / episode", str(max_steps))
    table.add_row("Mean return (first 10 eps)", f"{first_10:.1f}")
    table.add_row("Mean return (last 10 eps)", f"{last_10:.1f}")
    table.add_row("Improvement", f"{improvement:+.1f}%")
    table.add_row("Best episode return", f"{max(episode_returns):.1f}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE agent")
    parser.add_argument("--skus", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        n_skus=args.skus,
        n_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
    )