# Architecture Overview

This document explains how the RL Agent Plugin is structured — from the Godot scene graph down to gradient updates and distributed workers.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Godot Editor                           │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ RLDashboard │   │ RLSetupDock  │   │ Model Exporter  │  │
│  └──────┬──────┘   └──────────────┘   └─────────────────┘  │
│         │ reads metrics.jsonl                               │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         │          Training Process                         │
│  ┌──────▼──────────────────────────────────┐               │
│  │           TrainingBootstrap              │               │
│  │  ┌──────────┐  ┌───────────────────┐    │               │
│  │  │ RLAcademy│  │  Policy Groups    │    │               │
│  │  └──────────┘  │  ┌─────────────┐  │    │               │
│  │                │  │  PpoTrainer │  │    │               │
│  │  ┌──────────┐  │  │  SacTrainer │  │    │               │
│  │  │ RLAgent  │  │  └─────────────┘  │    │               │
│  │  │  2D/3D   │  └───────────────────┘    │               │
│  │  └──────────┘                           │               │
│  │       ▲ (N × BatchSize instances)        │               │
│  └───────┼──────────────────────────────────┘               │
│          │                                                  │
│  ┌───────┼──────────────────────────────────┐               │
│  │       │   DistributedMaster (TCP :7890)   │               │
│  └───────┼──────────────────────────────────┘               │
│          │ rollouts ↑  weights ↓                            │
│  ┌───────┼────────────────────────────────────────────┐     │
│  │  Worker 1   Worker 2   Worker 3  ...  Worker N     │     │
│  │  (headless, 4× sim speed)                          │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Scene Graph

A training scene has this structure:

```
TrainingScene (Node)
├── RLAcademy              ← coordinates the scene; holds all config resources
│   └── [config resources attached in Inspector]
└── Environment (Node2D/Node3D)
    ├── Agent_0 (RLAgent2D or RLAgent3D)
    ├── Agent_1 ...
    └── Arena/Obstacles/...
```

`TrainingBootstrap` (launched by **Start Training** from the toolbar or RL Setup dock) instantiates your scene **BatchSize times** side-by-side, so 4 parallel environments run simultaneously inside one process.

---

## Core Components

### RLAcademy

`RLAcademy` is the scene-level coordinator. It:

- Holds all configuration resources (algorithm, run, curriculum, distributed, self-play).
- Discovers every `RLAgent2D`/`RLAgent3D` in its subtree.
- Loads `.rlmodel` inference assets and creates `IInferencePolicy` instances when in **Inference** mode.
- Broadcasts curriculum progress to every agent each step.
- Hosts the spy overlay (debug visualization of observations/rewards/actions at runtime).

You configure training entirely through the Inspector on this node.

---

### TrainingBootstrap

`TrainingBootstrap` is the main training loop. It runs as a separate Godot scene, instantiating your training scene N times. Each physics frame:

1. **Value estimation** — call `EstimateValue()` on each agent's policy (needed for PPO's GAE advantage calculation).
2. **Episode resets** — call `OnEpisodeBegin()` on done agents; write episode metrics.
3. **Action sampling** — call `SampleAction()` to get the policy's stochastic action + log-probability.
4. **Apply decisions** — call `OnActionsReceived()` so the agent moves in the simulation.
5. **Step** — call `OnStep()` so the agent accumulates rewards and calls `EndEpisode()` when done.
6. **Record transitions** — store `(obs, action, reward, done, next_obs)` in the trainer's buffer.
7. **Update** — when the buffer is full, run backpropagation (synchronously or on a background thread).
8. **Checkpoint** — save weights to disk every N updates.

---

### Trainers (PpoTrainer / SacTrainer)

Each **policy group** gets its own trainer instance. A policy group is a set of agents that share the same neural network — identified by `PolicyGroupConfig.AgentId`.

The trainer owns:

- The neural network (forward + backward pass).
- The rollout buffer (PPO) or replay buffer (SAC).
- Gradient update logic.
- Checkpoint serialization.

Trainers implement three interfaces:

| Interface | Purpose |
|-----------|---------|
| `ITrainer` | Core: sample, record, update, checkpoint |
| `IAsyncTrainer` | Optional: background-thread gradient updates |
| `IDistributedTrainer` | Optional: receive worker rollouts, broadcast weights |

---

### Agents (RLAgent2D / RLAgent3D)

Agents are Godot nodes that you subclass. They implement four lifecycle methods:

```
OnEpisodeBegin()          ← reset the scene
CollectObservations()     ← fill the observation vector
OnActionsReceived()       ← apply the action to the simulation
OnStep()                  ← compute rewards, call EndEpisode() if done
```

And optionally:

```
DefineActions()           ← declare the action space (called once)
OnTrainingProgress()      ← receive curriculum progress [0, 1]
OnHumanInput()            ← read player input in Human control mode
```

See [configuration.md](configuration.md) for the full agent API.

---

### Distributed Training

When `RLDistributedConfig.WorkerCount > 0`, the plugin launches N headless Godot processes ("workers") alongside the main training process ("master").

```
Master process
  ├── Runs TrainingBootstrap with DistributedMaster node
  ├── TCP server on port 7890
  └── Owns the trainer (neural network weights)

Worker process × N
  ├── Runs the same training scene (headless, no display)
  ├── Connects to master via TCP
  ├── Collects rollouts at 4× simulation speed
  └── Sends transitions to master; receives updated weights
```

**Data flow:**

1. Workers collect experience and send `ROLLOUT` messages to the master.
2. The master injects worker transitions into its trainer buffer.
3. When the buffer is full, the master runs a gradient update.
4. The master broadcasts new weights (`WEIGHTS` message) to all workers.
5. Workers apply the new weights and keep collecting.

This architecture keeps training on the master GPU/CPU while workers focus purely on data collection.

---

### Inference

For deployment (or testing during development), set `ControlMode = Inference` on your agent and provide a `.rlmodel` path. The plugin loads the model and creates an `IInferencePolicy` that runs deterministic forward passes (no gradient tracking, no buffer writes).

```
RLAcademy (Inference mode)
  └── Agent
        └── IInferencePolicy
              ├── PpoInferencePolicy  ← argmax or tanh(mean) for PPO
              └── SacInferencePolicy  ← tanh(mean) for SAC
```

---

### Neural Network Architecture

Both PPO and SAC use feed-forward networks defined by `RLNetworkGraph`.

**PPO — PolicyValueNetwork:**

```
Observations (flat float[])
    │
    ▼
Trunk (shared layers)
  Dense(64, Tanh) → Dense(64, Tanh)
    │
    ├─► Policy Head
    │     Discrete:   Linear → Softmax → sample action
    │     Continuous: Linear → [mean, logStd] → Gaussian sample + tanh squash
    │
    └─► Value Head
          Linear → scalar (state value V(s))
```

**SAC — SacNetwork:**

```
Observations (flat float[])
    │
    ▼
Actor Network (trunk + Gaussian head)
  → [mean, logStd] → sample + tanh → action
    │
    ├─► Critic 1  (obs + action → Q₁)
    └─► Critic 2  (obs + action → Q₂)     ← prevents overestimation
          Target Critics (polyak-averaged copies)
```

---

### Metrics & Dashboard

Every completed episode, `TrainingBootstrap` appends a JSON line to:

```
RL-Agent-Training/<RunId>/metrics.jsonl
```

Example line:

```json
{
  "episode_reward": 12.4,
  "episode_length": 318,
  "policy_loss": 0.042,
  "value_loss": 0.019,
  "entropy": 0.74,
  "total_steps": 128000,
  "episode_count": 1024,
  "policy_group": "agent_0",
  "curriculum_progress": 0.35
}
```

`RLDashboard` polls this file every 2 seconds and renders live charts. Charts include: episode reward, episode length, policy loss, value loss, entropy, Elo (self-play), and curriculum progress.

---

## Data Flow Summary

```
Physics Frame
─────────────
Agent.CollectObservations()
    └─► ObservationBuffer → float[]

Trainer.SampleAction(obs)
    └─► PolicyNetwork forward pass → action + log_prob

Agent.OnActionsReceived(action)
    └─► simulation update (physics, velocity, etc.)

Agent.OnStep()
    └─► AddReward() / EndEpisode()

Trainer.RecordTransition(obs, action, reward, done, next_obs)
    └─► PPO: append to rollout buffer
        SAC: append to replay buffer

[When buffer full]
Trainer.TryUpdate()
    └─► Compute gradients → update network weights
        Write checkpoint
        Broadcast to workers (if distributed)
```
