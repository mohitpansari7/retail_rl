# RetailRL — Complete Run Guide

## 1. One-time setup

```bash
# Clone your repo and enter the project
cd retail_rl

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Run the full test suite (verify everything works first)

```bash
# From inside retail_rl/ directory
pytest tests/ -v
```

Expected: **154 tests pass** (torch-dependent tests need torch installed).

Run only the pure-Python tests if torch is slow to install:
```bash
pytest tests/test_env.py tests/test_ppo.py tests/test_reward_model.py \
       tests/test_llm_agent.py tests/test_grpo.py tests/test_safety.py \
       tests/test_deployment.py -v
```

---

## 3. Phase 1 — Smoke test (MDP environment)

Runs one episode with a random policy. Validates the environment is wired correctly.

```bash
python main.py --phase 1 --skus 20 --steps 30
```

What you should see:
- Step-by-step revenue output
- Final summary table with total revenue, stockouts
- "Phase 1 environment verified" message

---

## 4. Phase 2 — Train REINFORCE agent

```bash
# Quick run (verify it works)
python -m training.train_reinforce --skus 10 --episodes 50 --steps 30

# Full training run
python -m training.train_reinforce --skus 20 --episodes 200 --steps 60
```

What to watch:
- `Avg(10)` return should trend upward (noisy but directional)
- `π loss` should decrease over time
- Checkpoint saved to `checkpoints/reinforce_final.pt`

---

## 5. Phase 3 — Train PPO agent

```bash
# Quick run
python -m training.train_ppo --skus 10 --steps 10240 --rollout 512

# Full run
python -m training.train_ppo --skus 20 --steps 40960 --rollout 2048
```

What to watch vs Phase 2:
- PPO converges faster and with less variance in rewards
- `mean_ratio` should stay close to 1.0 (clipping is working)
- `entropy` should slowly decrease as policy becomes more confident

---

## 6. Phase 4 — Verify reward model components

No standalone training script (reward model trains as part of RLHF loop).
Run the tests to verify the math:

```bash
pytest tests/test_reward_model.py -v
```

To see the Elo tracker in action interactively:
```bash
python -c "
from reward.elo_rating import EloTracker
t = EloTracker()
t.register('ppo_v1')
t.register('ppo_v2')
t.record_match_by_returns('ppo_v1', 'ppo_v2', return_a=1200, return_b=1000)
t.record_match_by_returns('ppo_v1', 'ppo_v2', return_a=1100, return_b=1050)
print(t.summary())
"
```

---

## 7. Phase 5 — Multi-store environment

```bash
# Verify the multi-store env works
python -c "
from envs.multi_store_env import MultiStoreEnv
env = MultiStoreEnv(n_stores=4, n_skus=10, max_steps=5)
obs = env.reset(seed=42)
print('Stores:', env.store_ids)
print('Obs dim per store:', env.obs_dim)
print('Global state dim:', env.global_obs_dim)
for step in range(5):
    actions = env.sample_actions()
    obs, ind, joint, done, info = env.step(actions)
    print(f'Step {step+1}: joint_reward={joint:.1f}, price_variance={info[\"price_variance\"]:.3f}')
"
```

---

## 8. Phase 6 — CTDE / MAPPO

Run the CTDE buffer tests (fully verifiable without torch):
```bash
pytest tests/test_ctde.py -v
```

To train MAPPO (requires torch):
```bash
python -c "
from envs.multi_store_env import MultiStoreEnv
from agents.mappo_agent import MAPPOAgent
from config.settings import TrainingConfig

env = MultiStoreEnv(n_stores=4, n_skus=5, max_steps=20)
obs = env.reset(seed=42)

agent = MAPPOAgent(
    store_ids=env.store_ids,
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    global_state_dim=env.global_obs_dim,
    n_steps=40,
)
print('MAPPOAgent initialised OK')
print(f'Actors: {len(agent.actors)}, Critic: shared')

# One rollout
for step in range(40):
    actions = agent.select_actions(obs)
    agent.store_transition(obs, actions, env.get_global_state(), 100.0, False)
    obs, _, joint, done, info = env.step(actions)
    if done: obs = env.reset()

metrics = agent.update(last_global_state=env.get_global_state())
print('Update metrics:', metrics)
"
```

---

## 9. Phase 7 — LLM-augmented hierarchical agent

```bash
# Mock backend (fast, free — use for training)
python -m training.train_hierarchical --stores 4 --skus 5 --steps 2048 --rollout 256 --llm mock

# Full training
python -m training.train_hierarchical --stores 4 --skus 10 --steps 20480 --rollout 512 --llm mock
```

To use the real Claude API (costs credits — use for evaluation only):
```bash
# Set your API key first
export ANTHROPIC_API_KEY=your_key_here

python -m training.train_hierarchical --stores 4 --skus 5 --steps 512 --rollout 256 --llm anthropic
```

What to watch:
- `LLM calls: X` in the progress — should be ~steps/24 (once per simulated day)
- `Last price_aggression` in summary — should respond to festive season (Q4)

---

## 10. Phase 8 — GRPO + verifiable rewards

```bash
# Run verifiable reward checks interactively
python -c "
from reward.verifiable_rewards import (
    check_margin_constraint, check_stockout_constraint,
    check_revenue_target, VerifiableRewardScorer
)

# Simulate a good pricing decision
results = [
    check_margin_constraint(price=110, cost=100, sku_id='elec_001'),
    check_stockout_constraint(inventory=50, units_demanded=30),
    check_revenue_target(actual_revenue=12000, target_revenue=10000),
]
scorer = VerifiableRewardScorer()
score = scorer.score(results)
print(f'Score: {score:.3f}  (1.0 = all constraints satisfied)')
for r in results:
    print(f'  {r.name}: {\"PASS\" if r.passed else \"FAIL\"} — {r.detail}')
"
```

---

## 11. Phase 9 — Safety & governance

```bash
# Run the Lagrangian optimizer interactively
python -c "
from safety.constraints import LagrangianConstraintOptimizer

opt = LagrangianConstraintOptimizer()
print('Initial lambdas:', opt.lambdas)

# Simulate stockout violations
for _ in range(10):
    opt.update_lambdas({'stockout_rate': 0.08, 'price_variance': 0.05})

print('After 10 steps of stockout violations:')
print('Lambdas:', {k: round(v, 3) for k, v in opt.lambdas.items()})
print('Active constraints (λ>1):', opt.get_active_constraints())
"

# Run hacking detector
python -c "
import numpy as np
from safety.reward_hacking import RewardHackingDetector

detector = RewardHackingDetector()

# Simulate boundary exploitation — agent hits 15% every day
for _ in range(15):
    old_p = np.array([100.0] * 10)
    new_p = old_p * 1.14   # 14% change daily
    detector.record_price_changes(old_p, new_p)

alerts = detector.check_all()
print(f'Alerts raised: {len(alerts)}')
for a in alerts:
    print(f'  [{a.level.value.upper()}] {a.hack_type}: {a.detail}')
    print(f'  Recommendation: {a.recommended_action}')
"
```

---

## 12. Phase 10 — Deployment checks

```bash
# A/B test framework
python -c "
import numpy as np
from evaluation.deployment import ABTestFramework, ABGroup

ab = ABTestFramework(treatment_pct=0.10, minimum_episodes=20)
rng = np.random.default_rng(42)

# Simulate 60 episodes — treatment is 15% better
for _ in range(60):
    ab.record_episode(ABGroup.CONTROL,   float(rng.normal(1000, 50)))
    ab.record_episode(ABGroup.TREATMENT, float(rng.normal(1150, 50)))

result = ab.evaluate()
print(f'Control mean:    ₹{result.mean_control:,.0f}')
print(f'Treatment mean:  ₹{result.mean_treatment:,.0f}')
print(f'Treatment effect:₹{result.treatment_effect:,.0f}')
print(f'p-value:         {result.p_value:.4f}')
print(f'Significant:     {result.significant}')
print(f'Action:          {result.recommended_action}')
"

# Drift detection
python -c "
import numpy as np
from evaluation.deployment import DriftDetector, DriftLevel

# Baseline from training
baseline = np.random.default_rng(0).normal(1000, 50, 200).tolist()
detector = DriftDetector(baseline_rewards=baseline)

# Simulate competitor disruption — rewards drop significantly
for r in np.random.default_rng(1).normal(700, 50, 50):
    detector.record(float(r))

report = detector.check_drift()
print(f'Drift level: {report.level.value.upper()}')
print(f'Reward drift score: {report.psi_reward:.3f} (>2.0 = alert)')
print(f'Recommendation: {report.recommendation}')
"
```

---

## 13. Generate production report

```bash
python -c "
import numpy as np
from evaluation.metrics import ProductionMetricsTracker, EpisodeMetrics

tracker = ProductionMetricsTracker(store_ids=[f'store_{i}' for i in range(4)])
rng = np.random.default_rng(42)

# Simulate 30 episodes
for i in range(30):
    tracker.record_episode(EpisodeMetrics(
        episode=i,
        total_revenue=float(rng.normal(50000 + i*200, 2000)),
        stockout_count=int(rng.integers(0, 3)),
        margin_violations=0,
        policy_loss=float(max(0.01, 0.5 - i*0.015)),
        entropy=float(rng.uniform(0.6, 1.0)),
        constraint_violations=int(rng.integers(0, 2)),
    ))

print(tracker.generate_report())
"
```

---

## Quick reference — all training commands

```bash
# Phase 2
python -m training.train_reinforce --skus 20 --episodes 200 --steps 60

# Phase 3
python -m training.train_ppo --skus 20 --steps 40960 --rollout 2048

# Phase 7 (hierarchical)
python -m training.train_hierarchical --stores 4 --skus 10 --steps 20480 --llm mock

# All tests
pytest tests/ -v

# Tests by phase
pytest tests/test_env.py -v           # Phase 1
pytest tests/test_reinforce.py -v     # Phase 2
pytest tests/test_ppo.py -v           # Phase 3
pytest tests/test_reward_model.py -v  # Phase 4
pytest tests/test_marl.py -v          # Phase 5
pytest tests/test_ctde.py -v          # Phase 6
pytest tests/test_llm_agent.py -v     # Phase 7
pytest tests/test_grpo.py -v          # Phase 8
pytest tests/test_safety.py -v        # Phase 9
pytest tests/test_deployment.py -v    # Phase 10
```

---

## Expected directory structure after first full run

```
retail_rl/
├── checkpoints/
│   ├── reinforce_final.pt
│   ├── ppo_final.pt
│   └── hierarchical_mappo/
│       ├── actor_store_0.pt
│       ├── actor_store_1.pt
│       └── critic_shared.pt
├── logs/              (created automatically)
└── ...
```