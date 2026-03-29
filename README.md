# RL Agent Godot Workspace

This repository is the Godot project workspace used to run demos, iterate on scenes, and integrate addons.

The RL plugin itself now lives in its own repository:

- `addons/rl-agent-plugin` (Git submodule): <https://github.com/dron3flyv3r/rl-agent-plugin>

## What is in this repo

- Demo environments for single-agent, self-play, curriculum, and locomotion workflows
- Exported `.rlmodel` assets used by demo scenes
- Godot project files and integration glue around the standalone plugin repo

## Repository split

- `rl-agent-godot` (this repo): demo project, sample scenes, and integration
- `rl-agent-plugin` (submodule): RL runtime, editor tooling, training/inference implementation, plugin docs

## Getting started

Choose one setup path:

### Option 1: Release version (recommended)

1. Go to the [rl-agent-plugin](https://github.com/dron3flyv3r/rl-agent-plugin) repository.
2. Download and extract `rl-agent-plugin` release `v0.1.0-beta` from GitHub.
3. Place the extracted plugin at:

- `addons/rl-agent-plugin`

4. Open the project in Godot 4.6+ with C# support.
5. Enable **RL Agent Plugin** in **Project Settings -> Plugins**.
6. Build once with `Alt+B`.
7. Restart the editor if RLDash windows don't appear.


### Option 2: Clone
1a. Clone with submodules:

``` bash
git clone --recurse-submodules https://github.com/dron3flyv3r/rl-agent-godot.git
```

1b. If already cloned, initialize submodules:

``` bash
git submodule update --init --recursive
git -C addons/rl-agent-plugin fetch --tags
git -C addons/rl-agent-plugin checkout <release-tag>
```

Example:

``` bash
git -C addons/rl-agent-plugin fetch --tags
git -C addons/rl-agent-plugin checkout v0.1.0-beta
```

2. Open this folder in Godot 4.6+ with C# support.
3. Enable **RL Agent Plugin** in **Project Settings -> Plugins**.
4. Build once with `Alt+B`.
5. Restart the editor if RLDash windows don't appear.


## Demo quick run

1. Open `demo/01 SingleAgent/ReachTargetDemo.tscn`.
2. Click `Start Training` from the top toolbar (or `RL Setup` dock).
3. Open `RLDash` to monitor reward/loss/entropy.

## Documentation

- Workspace docs: `docs/README.md`
- Demo catalog: `docs/demos.md`
- Plugin docs: `addons/rl-agent-plugin/docs/README.md`
- Plugin repo README: `addons/rl-agent-plugin/README.md`

## License

MIT (workspace content). The plugin submodule has its own license and release cadence.
