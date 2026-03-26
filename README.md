# RL Agent Plugin for Godot 4

A reinforcement learning plugin for Godot 4 with in-editor training, live metrics, curriculum learning, self-play, and `.rlmodel` export for deployment.

---

## What this plugin gives you

- Train agents in-editor with **PPO** or **SAC**
- Use the **Start Training** button in the top toolbar or in the **RL Setup** dock
- Monitor live reward/loss/entropy charts in **RLDash**
- Scale data collection with distributed worker processes
- Build 2D/3D agents with discrete or continuous actions
- Export trained policies as portable **`.rlmodel`** files

---

## Requirements

- Godot **4.3+** with C#/.NET support
- .NET SDK **8.0+**

---

## Installation

1. Copy `addons/rl_agent_plugin` into your project's `addons/` folder.
2. In Godot: **Project → Project Settings → Plugins**.
3. Enable **RL Agent Plugin**.
4. Build once: **Build → Build Solution** (`Alt+B`).

After enabling the plugin you should see:
- **RLDash** as an editor main screen tab.
- **RL Setup** dock on the right side.
- Toolbar buttons: **Start Training**, **Stop Training**, and **Run Inference**.

---

## Fast start

For a full beginner walkthrough (including a Demo 04-style cube-to-target example), read **[Get Started Guide](docs/get-started.md)**.

Quick version:

1. Open `demo/01 SingleAgent/ReachTargetDemo.tscn`.
2. In the top toolbar, click **Start Training**.
3. Open **RLDash** to monitor reward/loss charts.
4. Export the trained run to `.rlmodel` from RLDash (**Export Run** or checkpoint-row **Export**).
5. Assign the exported model path to your agent's `PolicyGroupConfig.InferenceModelPath`.
6. Click **Run Inference**.

---

## Documentation map

### Start here
- **Get Started Guide**: `docs/get-started.md`
- **Demos overview**: `docs/demos.md`

### Core concepts
- **Architecture**: `docs/architecture.md`
- **Algorithms (PPO/SAC)**: `docs/algorithms.md`
- **Configuration reference**: `docs/configuration.md`
- **Tuning guide**: `docs/tuning.md`

### Recommended reading order for new users
1. `docs/get-started.md`
2. `docs/demos.md`
3. `docs/configuration.md`
4. `docs/tuning.md`
5. `docs/architecture.md`

---

## Project layout

```text
addons/rl_agent_plugin/
├── Editor/            # RLDash, RL Setup dock, model import/export
├── Resources/         # Config, model graph, schedules
└── Runtime/           # Agents, training, inference, distributed runtime
```

---

## License

MIT
