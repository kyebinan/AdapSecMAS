# AdapSecMAS

> **Adaptive Security for Multi-Embedded Agent Systems**  
> Yohan Emmanuel Binan — Master in Machine Learning, KTH Royal Institute of Technology  
> Supervisor: LCIS — Université Grenoble Alpes

---

## Overview

AdapSecMAS is a multi-agent reinforcement learning simulator for adaptive network security in embedded autonomous systems. Twenty networked agents learn to defend against three simultaneous network attacks — jamming, flooding, and spoofing — while coordinating through decentralised security protocols.

The system is split into two phases:

- **Phase 1 — MARL training** : headless network simulation, agents learn when to trigger security protocols
- **Phase 2 — Production** : trained policy deployed on a city delivery scenario with trucks and drones, rendered in real time with pygame

---

## Game Model

```
G = ⟨ I, S, {Aᵢ}, T, R_team, {Oᵢ}, γ=0.99 ⟩

I        = {1…20} agents — parameter sharing, common policy π
S        = S_net × S_att  (jammer field J(x,y,t), SNR_ij, queues, attack modes)
Aᵢ       = {0…6}  7 discrete actions
oᵢ_t     ∈ ℝ¹²   (7D own obs + 5D aggregated peer messages)
R_team   = M − C_sec − C_act
Algorithm = MAPPO + CTDE + GRU actor + centralised critic
```

### Actions

| id | Action | Against |
|----|--------|---------|
| 0 | noop | — |
| 1 | boost ×8 | jamming |
| 2 | boost ×16 | jamming |
| 3 | rate-limit | flooding |
| 4 | quarantine | flooding |
| 5 | verify-nonce | spoofing |
| 6 | trigger protocol | all |

### Security protocols (deterministic — RL decides when)

| Protocol | Against | Mechanism |
|----------|---------|-----------|
| FREQ-HOP | Jamming | Local majority vote → channel switch |
| BAN-VOTE | Flooding | Byzantine quorum ⌈(n+1)/2⌉ → peer ban |
| ID-REVOKE | Spoofing | Strong quorum ⌈2n/3⌉ → distributed CRL |

---

## Architecture

```
adaptsecmas/
│
├── core/               # Abstractions (interfaces, constants, message, metrics)
├── network/            # WirelessChannel, GossipMediator, ValidationPipeline
├── attackers/          # JammerAgent, FloodAgent, SpoofAgent, Factory
├── protocols/          # FreqHop, BanVote, IdRevoke, Factory
├── security/           # SecurityLevelMachine, RewardComputer
├── agents/             # Actor (GRU), Critic, RolloutBuffer, MAPPOTrainer
├── simulation/         # NetworkEnv — POSG Facade
├── observers/          # CSVLogger, ConsoleLogger, PlotCollector
├── config/             # EnvConfig, MARLConfig, AttackConfig
│
├── train.py            # Headless training loop
├── eval.py             # 3-scenario evaluation
├── run_production.py   # Production visualisation entry point
│
└── production/         # DeliveryEnv, MapView, NetworkView
```

### Design patterns used

| Pattern | Where |
|---------|-------|
| Strategy | `IAttacker`, `IProtocol` |
| State | `SecurityLevelMachine` |
| Observer | `IMetricsObserver` → CSVLogger, ConsoleLogger, PlotCollector |
| Facade | `NetworkEnv` |
| Factory | `AttackerFactory`, `ProtocolFactory` |
| Mediator | `GossipMediator` |
| Chain of Responsibility | `ValidationPipeline` |
| Template Method | `BaseAttacker`, `BaseProtocol` |

---

## Installation

```powershell
# Windows — create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA (RTX 3060 / CUDA 12.5)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
pip install numpy scipy matplotlib pandas gymnasium pytest pytest-cov pygame pyyaml tqdm
```

Verify GPU detection:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Usage

### 1. Train

```powershell
python train.py

# Options
python train.py --episodes 300 --seed 42
python train.py --no-curriculum        # all attacks from episode 1
python train.py --steps-per-episode 1024
```

Curriculum:
- Episodes 0–49 : jamming only
- Episodes 50–119 : jamming + flooding
- Episodes 120+ : all 3 attacks

Outputs: `checkpoints/best.pt`, `logs/train_episodes.csv`, `logs/train_steps.csv`

### 2. Evaluate

```powershell
python eval.py --weights checkpoints/best.pt

# Options
python eval.py --weights checkpoints/best.pt --episodes 30 --seed 0
```

Three scenarios:
- `sim1_baseline` : no attacks, no defence
- `sim2_attack` : all attacks, no defence
- `sim3_defended` : all attacks, MARL active

Outputs: `logs/eval_summary.csv`, `logs/eval_episodes.csv`

### 3. Production visualisation

```powershell
python run_production.py --weights checkpoints/best.pt

# Options
python run_production.py --weights checkpoints/best.pt --fps 30
```

Two pygame windows open side by side:
- **Window 1** — city map with jammer heatmap, trucks, drones, packages
- **Window 2** — network dashboard with SNR links, queue bars, security levels

---

## Reward function

```
R_team = M − C_sec − C_act

M      = +2.0 × Δdelivery_rate
         +0.1 × n_links_healthy
         +1.0 × Δprotocol_success

C_sec  = +1.0 × n_msgs_lost_to_jam
         +0.8 × n_queue_overflows
         +1.5 × n_spoof_accepted      ← highest: identity attack cascades
         +2.0 × level_mismatch        ← primary escalation signal
         +0.5 × n_protocol_failed

C_act  = cost(action, security_level) ← noop costs 0.20 at CRITICAL
```

---

## Related work

- **Baudet (2023)** — MAKI: decentralised key infrastructure for MEAS. ID-REVOKE protocol directly extends MAKI's distributed CRL.
- **Albrecht, Christianos & Schäfer** — *Multi-Agent Reinforcement Learning*. POSG formalism, CTDE, MAPPO.
- **Yu et al. (2022)** — MAPPO: on-policy multi-agent PPO with centralised value function.

---

## Licence

MIT