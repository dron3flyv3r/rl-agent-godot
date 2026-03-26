# RL Agent Plugin for Godot 4

A reinforcement learning training and inference framework for Godot 4. Train agents using PPO or SAC directly inside the Godot editor with live dashboards, distributed multi-process training, curriculum learning, and self-play.

![RL Dashboard overview screenshot](docs/images/dashboard_overview.png)
> **[Image placeholder]** Screenshot of the RLDashboard dock in the Godot editor showing live training charts.

---

## Features

- **Two algorithms**: PPO (on-policy) and SAC (off-policy)
- **Distributed training**: Scale to N headless worker processes for faster data collection
- **Curriculum learning**: Automatically ramp task difficulty as the agent improves
- **Self-play**: Train competitive agents against historical snapshots of themselves
- **Live dashboard**: Real-time reward/loss/entropy charts inside the Godot editor
- **2D and 3D agents**: `RLAgent2D` and `RLAgent3D` base classes
- **Discrete and continuous action spaces**
- **Custom network architectures** via `RLNetworkGraph`
- **Hyperparameter schedules**: Decay learning rate, entropy coefficient, or clip epsilon over training

---

## Installation

### Requirements

- Godot 4.3+ with .NET / C# support enabled
- .NET SDK 8.0 or later

### Steps

1. Copy the `addons/rl_agent_plugin` folder into your project's `addons/` directory.
2. In Godot, open **Project → Project Settings → Plugins** and enable **RL Agent Plugin**.
3. Build the C# solution: **Build → Build Solution** (or press `Alt+B`).
4. The **RL Dashboard** dock will appear at the bottom of the editor.

```
your_project/
├── addons/
│   └── rl_agent_plugin/   ← copy this folder here
├── scenes/
└── ...
```

---

## Quick Start: Train Your First Agent

### 1. Create a training scene

Add the following nodes to your scene:

```
TrainingScene (Node)
├── RLAcademy          ← scene coordinator
└── AgentEnv (Node2D)
    └── MyAgent        ← your agent script
```

### 2. Write your agent

```csharp
using Godot;
using RLAgentPlugin.Runtime.Agents;
using RLAgentPlugin.Runtime.Actions;
using RLAgentPlugin.Runtime.Observations;

public partial class MyAgent : RLAgent2D
{
    [Export] public Node2D Target;

    protected override void DefineActions(ActionSpaceBuilder builder)
    {
        builder.AddDiscrete("Move", "Left", "Right", "Idle");
    }

    public override void CollectObservations(ObservationBuffer obs)
    {
        obs.AddNormalized(GlobalPosition.X, -500f, 500f);
        obs.AddNormalized(Target.GlobalPosition.X, -500f, 500f);
    }

    protected override void OnActionsReceived(ActionBuffer actions)
    {
        int move = actions.GetDiscrete("Move");
        if (move == 0) Position += Vector2.Left * 100f * (float)GetPhysicsProcessDeltaTime();
        if (move == 1) Position += Vector2.Right * 100f * (float)GetPhysicsProcessDeltaTime();
    }

    protected override void OnStep()
    {
        float dist = GlobalPosition.DistanceTo(Target.GlobalPosition);
        AddReward(-dist * 0.001f);     // penalty for distance
        if (dist < 20f)
        {
            AddReward(1f);             // bonus for reaching target
            EndEpisode();
        }
        if (EpisodeSteps > 500) EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        Position = new Vector2(GD.RandRange(-200f, 200f), 0f);
    }
}
```

### 3. Configure the Academy

Select the `RLAcademy` node in the Inspector and set:

| Property | Value |
|----------|-------|
| Training Config → Algorithm | `RLPPOConfig` |
| Run Config → Batch Size | `4` |
| Max Episode Steps | `500` |

### 4. Launch training

Open **RL Dashboard** dock → click **Launch Training**.

Charts will populate as episodes complete. Checkpoints are saved to `RL-Agent-Training/<RunId>/`.

### 5. Run inference

```csharp
// In your agent, set the inference model path:
[Export] public string InferenceModelPath = "res://RL-Agent-Training/MyRun/best.rlmodel";
```

Or set `ControlMode = RLControlMode.Inference` on the agent and point `PolicyGroupConfig.InferenceModelPath` to the exported `.rlmodel` file.

---

## Documentation

| Topic | File |
|-------|------|
| Architecture overview | [docs/architecture.md](docs/architecture.md) |
| Algorithms (PPO & SAC) | [docs/algorithms.md](docs/algorithms.md) |
| Configuration reference | [docs/configuration.md](docs/configuration.md) |
| Tuning guide & tips | [docs/tuning.md](docs/tuning.md) |
| Demo environments | [docs/demos.md](docs/demos.md) |

---

## Project Layout

```
addons/rl_agent_plugin/
├── Editor/            # Dashboard, setup dock, model import/export
├── Resources/         # Config, network, schedule, checkpoint resources
│   ├── Config/        # RLPPOConfig, RLSACConfig, RLRunConfig, ...
│   ├── Models/        # RLNetworkGraph, RLLayerDef
│   └── Schedules/     # RLHyperparamSchedule subclasses
└── Runtime/           # Core training & inference framework
    ├── Agents/        # RLAgent2D, RLAgent3D
    ├── Core/          # IRLAgent, RLAcademy, RLActionDefinition
    ├── Training/      # PpoTrainer, SacTrainer, self-play
    ├── Distributed/   # DistributedMaster, DistributedWorker
    └── Inference/     # IInferencePolicy, PPO/SAC inference wrappers
```

---

## License

MIT
