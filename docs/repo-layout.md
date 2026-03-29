# Repository Layout

This repo is a Godot workspace that consumes the RL plugin as a Git submodule.

## Top-level folders

- `addons/rl-agent-plugin/`
  - Separate Git repository (submodule)
  - Contains plugin runtime/editor code and plugin documentation
- `demo/`
  - Demo training scenes and scripts

## Working with submodules

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/dron3flyv3r/rl-agent-godot.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

Update plugin submodule to the latest released plugin version:

```bash
cd addons/rl-agent-plugin
git fetch --tags
git checkout v0.1.0-beta
cd ../..
git add addons/rl-agent-plugin
```

If you want bleeding-edge plugin changes instead, use `main`.

## Which repo should I edit?

Edit `rl-agent-godot` when changing:

- Demo scenes/scripts
- Workspace-level Godot project setup

Edit `rl-agent-plugin` when changing:

- `Runtime/`, `Editor/`, `Resources/`, `Scenes/Bootstrap/`
- RL algorithms, training loop, inference, checkpoint formats
- Plugin-facing docs and configuration reference
