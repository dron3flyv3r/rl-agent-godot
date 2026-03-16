# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RL Agent Plugin** is a Godot 4.6 addon (C# / .NET 8.0) that enables reinforcement learning training directly inside the Godot editor. Agents are trained using PPO or SAC algorithms in-process, with a live dashboard for monitoring.

## Build & Run

This is a Godot 4.6 C# project. There is no separate build step — Godot compiles C# on launch.

- **Open in editor**: `godot --path /home/kasper/gameProjects/rl-agent-plugin` (or open Godot and load the project)
- **Godot binary**: `/usr/lib/godot-mono/godot.linuxbsd.editor.x86_64.mono`
- **Start training**: Click "Start Training" in the Godot toolbar (requires a valid training scene open)
- **View metrics**: Open the "RLDash" tab in the editor bottom panel

Training output is written to `res://RL-Agent-Training/runs/{runId}/`.

## Architecture

### Core Data Flow

1. User clicks **Start Training** → `RLAgentPluginEditor` validates the scene and writes a `TrainingLaunchManifest` to user storage
2. Godot switches to `TrainingBootstrap.tscn`, which reads the manifest
3. `TrainingBootstrap` instantiates N copies of the training scene, creates `ITrainer` instances per policy group, and runs the training loop
4. Metrics (`metrics.jsonl`, `status.json`) are polled every 2s by `RLDashboard`

### Key Types

| Type | Location | Role |
|------|----------|------|
| `RLAgent2D` | `Runtime/` | Base class for all agents — subclass to add observations, actions, and rewards |
| `RLAcademy` | `Runtime/` | Node placed in the training scene; configures trainer/network settings and holds all agents |
| `TrainingBootstrap` | `Scenes/` | Orchestrates training runs (batching, checkpointing, metric writing) |
| `PpoTrainer` / `SacTrainer` | `Runtime/` | Learning algorithms; implement `ITrainer` |
| `PolicyValueNetwork` | `Runtime/` | Neural network (shared trunk + policy/value heads); handles its own weight save/load |
| `RLAgentPluginEditor` | `Editor/` | Plugin entry point; registers node types, toolbar buttons, and the dashboard |
| `RLDashboard` | `Editor/` | Live training monitor; polls metrics files and drives `LineChartPanel` charts |

### Agent API (how users extend the plugin)

Agents subclass `RLAgent2D` and:

1. **Define actions** via attributes:
   ```csharp
   [DiscreteAction(4, "up", "down", "left", "right")]
   public int Movement { get; set; }

   [ContinuousAction(2)]
   public float[] Steering { get; set; }
   ```

2. **Collect observations** by writing into an `ObservationBuffer`:
   ```csharp
   public override void CollectObservations(ObservationBuffer buffer)
   {
       buffer.Add(Position);            // Vector2 → 2 floats
       buffer.AddNormalized(hp, 0, 100);
   }
   ```

3. **Assign rewards** via `AddReward(float)` or `SetReward(float)` each physics step.

`RLActionBinding` uses reflection to discover action attributes and apply them automatically — no manual wiring needed.

### Control Modes

Agents have three modes (set on `RLAgent2D.ControlMode`):
- `Train` — policy is updated by the trainer
- `Inference` — runs the loaded model without gradient updates
- `Human` — bypasses the policy (for keyboard-controlled agents in demos)

### Configuration Resources

- `RLTrainerConfig` — hyperparameters (lr, gamma, rollout length, batch size, entropy coeff, etc.)
- `RLNetworkConfig` — architecture (hidden sizes, activation, optimizer)

Both are Godot `Resource` subclasses, editable in the Inspector on `RLAcademy`.

## Demo

`demo/TagDemo.tscn` is the reference multi-agent scene: two chasers vs. two runners. `TagArenaController.cs` manages episode resets, rewards, and boundary conditions. `TagAgent.cs` is a minimal `RLAgent2D` subclass showing how to implement the agent API.
