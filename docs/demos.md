# Demo Environments

All demos are in `demo/` and can be launched directly in this workspace.

## 01 SingleAgent

- Scene: `demo/01 SingleAgent/ReachTargetDemo.tscn`
- Script: `demo/01 SingleAgent/Scripts/ReachTargetAgent.cs`
- Focus: Basic 2D navigation with a single policy group

## 02 MultiAgentSelfPlay

- Scene: `demo/02 MultiAgentSelfPlay/TagDemo.tscn`
- Scripts: `TagAgent.cs`, `TagArenaController.cs`, player controllers
- Focus: Competitive multi-agent setup with self-play policy pairings

## 03 WallClimbCurriculum

- Scene: `demo/03 WallClimbCurriculum/WallClimbDemo.tscn`
- Scripts: `WallClimbAgent.cs`, `WallClimbArenaController.cs`
- Focus: Curriculum-based difficulty progression in 3D control

## 04 MoveToTarget3D

- Scene: `demo/04 MoveToTarget3D/MoveToTarget3D.tscn`
- Script folder: `demo/04 MoveToTarget3D/Scripts/`
- Focus: Simple continuous-control 3D task (good PPO/SAC comparison baseline)

## 05 Crawler

- Scene: `demo/05 Crawler/CrawlerDemo.tscn`
- Script folder: `demo/05 Crawler/Scripts/`
- Focus: Locomotion with higher-dimensional continuous actions

## Typical workflow

1. Open a demo scene.
2. Verify an `RLAcademy` node exists and has configs assigned.
3. Start training from toolbar (`Start Training`) or `RL Setup` dock.
4. Observe run metrics in `RLDash`.
5. Export checkpoints to `.rlmodel` and test inference.

For plugin internals and configuration parameter details, see:
- `addons/rl-agent-plugin/README.md`
- `addons/rl-agent-plugin/docs/README.md`
