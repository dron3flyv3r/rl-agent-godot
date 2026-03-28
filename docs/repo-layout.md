# Repository Layout

This repo is a Godot workspace that consumes the RL plugin as a Git submodule.

## Top-level folders

- `addons/rl-agent-plugin/`
  - Separate Git repository (submodule)
  - Contains plugin runtime/editor code and plugin documentation
- `demo/`
  - Demo training scenes and scripts
- `AgentExports/` and `Demo2AgentExports/`
  - Exported `.rlmodel` files used by demos

## Working with submodules

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/dron3flyv3r/rl-agent-godot.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

Update plugin submodule to latest main:

```bash
cd addons/rl-agent-plugin
git checkout main
git pull
cd ../..
git add addons/rl-agent-plugin
```

## Which repo should I edit?

Edit `rl-agent-godot` when changing:

- Demo scenes/scripts
- Workspace-level Godot project setup

Edit `rl-agent-plugin` when changing:

- `Runtime/`, `Editor/`, `Resources/`, `Scenes/Bootstrap/`
- RL algorithms, training loop, inference, checkpoint formats
- Plugin-facing docs and configuration reference
