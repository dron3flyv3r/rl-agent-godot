# Repository Guidelines

## Project Structure & Module Organization
`addons/rl_agent_plugin` contains the main Godot 4 C# plugin. Keep runtime gameplay code in `Runtime/`, editor-only tooling in `Editor/`, shared resources in `Resources/`, and bootstrap scenes in `Scenes/`. `addons/godot_mcp` is a separate GDScript editor plugin and should stay isolated from RL runtime changes. `demo/` holds playable sample scenes such as `TagDemo.tscn` and `ReachTargetDemo.tscn`. Training outputs land in `RL-Agent-Training/runs/`; exported models go in `AgentExports/`.

## Build, Test, and Development Commands
Use `dotnet build 'RL agent plugin.sln'` to compile the plugin and catch C# errors. Open the project in Godot 4.6 to run editor tooling and demos; if the CLI is installed, `godot-mono --path .` opens the project and `godot-mono --path . --scene res://demo/TagDemo.tscn` runs the main demo scene. There is no dedicated lint target in the repository today, so treat a clean build and a quick in-editor smoke test as the baseline check before submitting changes.

## Coding Style & Naming Conventions
C# code uses 4-space indentation, nullable reference types, `PascalCase` for public types and members, and `_camelCase` for private fields. Match the existing file layout: one top-level type per file, with filenames matching the class name, for example `RLAgent2D.cs`. GDScript files in `addons/godot_mcp` follow Godot style with tabs, snake_case functions, and explicit type annotations where already used. Keep serialized resource and scene names descriptive, not abbreviated.

## Testing Guidelines
There is no automated test project yet. Validate changes by building the solution, opening the affected demo scene, and exercising the related editor workflow or training path. When touching training, verify that fresh run data is written under `RL-Agent-Training/runs/` and that checkpoints or metrics files update as expected.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Refactor demo2 into a new Tag demo`. Follow that pattern and keep subjects under about 72 characters. Pull requests should explain the user-visible impact, list affected scenes or plugin areas, and include screenshots or short recordings for editor UI changes. Note any generated artifacts, sample data, or follow-up work explicitly.
