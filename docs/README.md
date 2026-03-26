# Documentation Index

## New users
1. [Get Started Guide](get-started.md)
2. [Demo Environments](demos.md)
3. [Configuration Reference](configuration.md)
4. [Get Started Tips & Tricks](get-started.md#tips--tricks)

## Deep dives
- [Algorithms (PPO & SAC)](algorithms.md)
- [Tuning Guide](tuning.md)
- [Architecture Overview](architecture.md)

## Common tasks
- **First agent structure**: keep movement on a player node and put `RLAgent2D/RLAgent3D` as a child agent node
- **Start training**: top toolbar **Start Training** or right-side **RL Setup** dock
- **Watch metrics**: open **RLDash**
- **Export model**: in RLDash, use **Export Run** (or checkpoint-row **Export**) to create `.rlmodel`
- **Run inference**: set `PolicyGroupConfig.InferenceModelPath` and click **Run Inference**
