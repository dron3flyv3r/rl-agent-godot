# Algorithm Bench File Plan

This folder is split into three layers so fast trainer-level checks do not get mixed together with full runtime/system measurements.

## L0: Microbench / Smoke

Purpose:
- native/runtime sanity checks
- direct trainer construction
- checkpoint roundtrip
- tiny toy-task smoke loops

Files:
- `../Main.cs`
- `../TestScene.tscn`
- `../BenchMain.cs`
- `../BenchScene.tscn`
- `AlgorithmBenchContracts.cs`
- `AlgorithmBenchCatalog.cs`
- `AlgorithmBenchEnvironments.cs`
- `AlgorithmBenchRunner.cs`

## L1: Synthetic Throughput

Purpose:
- code-only performance measurements
- batched actors with `SampleActions(VectorBatch)`
- small vs large networks
- light vs heavy synthetic environments
- update throughput for PPO, A2C, DQN, SAC
- planning throughput for MCTS

Files:
- `AlgorithmBenchPerformanceRunner.cs`
- `AlgorithmBenchCatalog.cs`
- `AlgorithmBenchEnvironments.cs`
- `AlgorithmBenchContracts.cs`

Design:
- stays code-only
- no `RLAcademy`, no scene-instanced env grid
- simulates many actors via arrays of `IAlgorithmBenchEnvironment`
- measures decision latency, update latency, env steps/sec, updates/sec

## L2: Full System Bench

Purpose:
- real `TrainingBootstrap`
- real `RLAcademy`
- real `BatchSize`
- real scene instancing costs
- real multi-agent/shared-policy wiring
- distributed master/worker timing

Planned files:
- `../SystemBenchScene.tscn`
- `../SystemBenchMain.cs`
- `SystemBenchScenarioCatalog.cs`
- `SystemBenchScenarioRunner.cs`
- `SystemBenchAgent.cs`
- `SystemBenchController.cs`
- `SystemBenchEnvironmentModel.cs`

Notes:
- distributed currently applies to PPO and SAC only because `IDistributedTrainer` is implemented there today
- DQN, A2C, and MCTS can still be covered in L2 single-process and batched scenarios
