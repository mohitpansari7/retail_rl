"""
training/train_hierarchical.py
────────────────────────────────
Phase 7 — Training loop for the hierarchical LLM + MAPPO agent.

Key difference from train_ppo.py and train_mappo.py:
  - Each step: check if LLM should update strategy → augment obs → MAPPO acts
  - env_state is built from info dict and passed to LLM tools each step
  - Buffer stores AUGMENTED observations (obs + strategy_vector)
  - LLM call count tracked separately from env steps

Training uses "mock" LLM backend (fast, free, deterministic).
Swap to "anthropic" backend for evaluation/deployment.

Usage:
    python -m training.train_hierarchical --skus 10 --steps 20000
    python -m training.train_hierarchical --skus 20 --steps 40000 --llm anthropic
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from envs.multi_store_env import MultiStoreEnv
from agents.hierarchical_agent import HierarchicalPricingAgent
from config.settings import TrainingConfig

console = Console()


def _build_env_state_from_info(info: Dict, step: int) -> Dict:
    """
    Build the env_state dict that LLM tools will query.

    Translates the raw info dict from MultiStoreEnv.step() into
    the structured format RetailTools expects.

    This is the bridge between environment output and LLM input.
    In production this would pull from a real data warehouse.
    """
    # Aggregate demand history across stores
    demand_history: Dict[str, list] = {
        "electronics": [], "groceries": [], "apparel": [], "home_kitchen": []
    }
    our_prices:  Dict[str, float] = {}
    comp_prices: Dict[str, float] = {}
    inventory:   Dict[str, float] = {}

    for sid, store_info in info.get("store_infos", {}).items():
        for sku_info in store_info.get("skus", []):
            cat = sku_info["sku_id"].split("_")[0]
            if cat in demand_history:
                demand_history[cat].append(sku_info.get("units_sold", 0.0))

            key = f"{sid}_{sku_info['sku_id']}"
            our_prices[cat]  = sku_info.get("price", 100.0)
            comp_prices[cat] = sku_info.get("price", 100.0) * np.random.uniform(0.9, 1.1)
            inventory[key]   = sku_info.get("inventory", 100.0)

    return {
        "day_of_year":            (step % 365) + 1,
        "demand_history":         demand_history,
        "our_avg_prices":         our_prices,
        "competitor_avg_prices":  comp_prices,
        "inventory_levels":       inventory,
        "joint_reward":           info.get("joint_reward", 0.0),
        "price_variance":         info.get("price_variance", 0.0),
    }


def train(
    n_stores: int = 4,
    n_skus: int = 10,
    total_steps: int = 20_480,
    n_steps_per_rollout: int = 512,
    strategy_interval: int = 24,
    llm_backend: str = "mock",
    seed: int = 42,
    save_dir: str = "checkpoints",
) -> HierarchicalPricingAgent:
    """
    Train the hierarchical LLM + MAPPO agent.

    Args:
        n_stores             : Number of QuickMart stores.
        n_skus               : SKUs per store (subset for speed).
        total_steps          : Total environment steps.
        n_steps_per_rollout  : MAPPO rollout length before each update.
        strategy_interval    : Steps between LLM strategy calls.
        llm_backend          : "mock" (training) or "anthropic" (eval).
        seed                 : Random seed.
        save_dir             : Checkpoint directory.

    Returns:
        Trained HierarchicalPricingAgent.
    """
    Path(save_dir).mkdir(exist_ok=True)
    cfg = TrainingConfig(seed=seed)

    # ── Environment
    env = MultiStoreEnv(
        n_stores=n_stores,
        n_skus=n_skus,
        max_steps=365,
        seed=seed,
    )
    obs = env.reset(seed=seed)

    # ── Hierarchical agent
    # obs_dim and global_state_dim use ORIGINAL dimensions —
    # HierarchicalPricingAgent handles augmentation internally.
    agent = HierarchicalPricingAgent(
        store_ids=env.store_ids,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_obs_dim,
        cfg=cfg,
        strategy_interval=strategy_interval,
        llm_backend=llm_backend,
        n_steps=n_steps_per_rollout,
        device=cfg.device,
    )

    console.print(f"\n[bold cyan]Phase 7: Hierarchical LLM + MAPPO Training[/bold cyan]")
    console.print(f"Stores: {n_stores}  |  SKUs: {n_skus}  |  Backend: {llm_backend}")
    console.print(f"obs_dim: {env.obs_dim}  →  aug_obs_dim: {agent.aug_obs_dim}")
    console.print(f"Strategy interval: every {strategy_interval} steps")
    console.print(f"Total steps: {total_steps:,}  |  Rollout: {n_steps_per_rollout}\n")

    episode_rewards: list = []
    ep_reward = 0.0
    update_metrics_log = []
    env_state: Dict = {}

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        task = progress.add_task("[green]Training...", total=total_steps)

        for step in range(total_steps):

            # ── Build env state for LLM tools
            # On first step env_state is empty — LLM uses defaults.
            # After first step it's populated from the last info dict.

            # ── Act: LLM may update strategy, MAPPO selects prices
            actions = agent.select_actions(obs, env_state=env_state)

            # ── Step environment
            next_obs, ind_rewards, joint_reward, done, info = env.step(actions)

            # ── Build env_state for NEXT step's LLM call
            env_state = _build_env_state_from_info(info, step)
            agent.update_env_state(env_state)

            # ── Store transition with augmented obs
            # Scale reward same as PPO: /1e7 + clip to prevent critic explosion
            joint_reward_scaled = float(np.clip(joint_reward / 1e7, -10.0, 10.0))
            agent.store_transition(
                local_obs=obs,
                actions=actions,
                global_state=env.get_global_state(),
                joint_reward=joint_reward_scaled,
                done=done,
            )
            ep_reward += joint_reward  # track raw for reporting

            # ── Handle episode end
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs = env.reset(seed=seed + step)
                env_state = {}
            else:
                obs = next_obs

            # ── MAPPO update when buffer full
            if agent.buffer_full():
                metrics = agent.update(last_local_obs=obs)
                update_metrics_log.append(metrics)

                if len(update_metrics_log) % 5 == 0:
                    recent = episode_rewards[-10:] if episode_rewards else [0]
                    progress.stop()
                    console.print(
                        f"  Update {metrics.get('update', '?'):>4} | "
                        f"Steps: {step:>8,} | "
                        f"LLM calls: {metrics.get('llm_calls_total', 0):>5} | "
                        f"Critic loss: {metrics.get('critic_loss', 0):>7.4f} | "
                        f"Avg reward: {np.mean(recent):>10.1f}"
                    )
                    progress.start()

            progress.advance(task)

    # ── Summary
    console.print("\n[bold green]Hierarchical Training complete![/bold green]")
    _print_summary(episode_rewards, update_metrics_log, agent)

    agent.mappo.save(os.path.join(save_dir, "hierarchical_mappo"))
    return agent


def _print_summary(
    episode_rewards: list,
    update_metrics: list,
    agent: HierarchicalPricingAgent,
) -> None:
    if not episode_rewards:
        console.print("[yellow]No complete episodes.[/yellow]")
        return

    first_10 = np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    last_10  = np.mean(episode_rewards[-10:])
    improvement = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0.0

    table = Table(title="Hierarchical Agent Training Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Total episodes",            str(len(episode_rewards)))
    table.add_row("Total MAPPO updates",       str(len(update_metrics)))
    table.add_row("Total LLM strategy calls",  str(agent._llm_calls))
    table.add_row("Mean reward (first 10)",    f"{first_10:.1f}")
    table.add_row("Mean reward (last 10)",     f"{last_10:.1f}")
    table.add_row("Improvement",               f"{improvement:+.1f}%")
    table.add_row("Augmented obs dim",         str(agent.aug_obs_dim))

    if update_metrics:
        last_m = update_metrics[-1]
        table.add_row("Final critic loss",     f"{last_m.get('critic_loss', 0):.4f}")
        table.add_row("Final entropy",         f"{last_m.get('mean_entropy', 0):.4f}")

    if agent.strategy_history:
        last_sv = agent.strategy_history[-1]["strategy"]
        table.add_row("Last price_aggression", f"{last_sv[0]:.3f}")
        table.add_row("Last stockout_priority",f"{last_sv[1]:.3f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hierarchical LLM+MAPPO agent")
    parser.add_argument("--stores",    type=int,   default=4)
    parser.add_argument("--skus",      type=int,   default=10)
    parser.add_argument("--steps",     type=int,   default=20_480)
    parser.add_argument("--rollout",   type=int,   default=512)
    parser.add_argument("--interval",  type=int,   default=24)
    parser.add_argument("--llm",       type=str,   default="mock",
                        choices=["mock", "anthropic"])
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    train(
        n_stores=args.stores,
        n_skus=args.skus,
        total_steps=args.steps,
        n_steps_per_rollout=args.rollout,
        strategy_interval=args.interval,
        llm_backend=args.llm,
        seed=args.seed,
    )