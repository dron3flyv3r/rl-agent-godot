using Godot;
using Godot.Collections;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo.Benchmarks;

public static class AlgorithmBenchCatalog
{
    public static AlgorithmBenchCase[] CreateDefault()
    {
        return new[]
        {
            new AlgorithmBenchCase
            {
                Id = "ppo_discrete_bandit_smoke",
                Name = "PPO / Discrete Bandit / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.DiscreteBandit,
                Algorithm = RLAlgorithmKind.PPO,
                IsImplemented = true,
                Notes = "Constructs PPO directly, runs a short one-step episodic loop, then checkpoint-resume smoke.",
                CreateEnvironment = static () => new DiscreteBanditBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.PPO,
                    RolloutLength = 32,
                    EpochsPerUpdate = 2,
                    PpoMiniBatchSize = 16,
                    LearningRate = 0.001f,
                    Gamma = 0.99f,
                    GaeLambda = 0.95f,
                    ClipEpsilon = 0.2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = true,
                    ValueClipEpsilon = 0.2f,
                    EntropyCoefficient = 0.01f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "a2c_discrete_bandit_smoke",
                Name = "A2C / Discrete Bandit / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.DiscreteBandit,
                Algorithm = RLAlgorithmKind.A2C,
                IsImplemented = true,
                Notes = "Single-pass actor-critic smoke on the same tiny discrete benchmark as PPO.",
                CreateEnvironment = static () => new DiscreteBanditBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.A2C,
                    RolloutLength = 32,
                    EpochsPerUpdate = 1,
                    PpoMiniBatchSize = 32,
                    LearningRate = 0.001f,
                    Gamma = 0.99f,
                    GaeLambda = 1.0f,
                    ClipEpsilon = float.MaxValue / 2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = false,
                    ValueClipEpsilon = 0f,
                    EntropyCoefficient = 0.01f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "dqn_line_world_smoke",
                Name = "DQN / Line World / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.DQN,
                StepBudget = 384,
                EpisodeBudget = 24,
                IsImplemented = true,
                Notes = "Exercises replay buffer, target network updates, and checkpoint-resume on a deterministic discrete task.",
                CreateEnvironment = static () => new DiscreteLineWorldBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.DQN,
                    LearningRate = 0.001f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 5f,
                    ReplayBufferCapacity = 4096,
                    DqnBatchSize = 32,
                    DqnWarmupSteps = 32,
                    DqnEpsilonStart = 1.0f,
                    DqnEpsilonEnd = 0.05f,
                    DqnEpsilonDecaySteps = 256,
                    DqnTargetUpdateInterval = 32,
                    DqnUseDouble = true,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "mcts_line_world_smoke",
                Name = "MCTS / Line World / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.MCTS,
                StepBudget = 128,
                EpisodeBudget = 16,
                IsImplemented = true,
                Notes = "Pure planning smoke. No learning expected; validates environment-model wiring and decision loop latency.",
                CreateEnvironment = static () => new DiscreteLineWorldBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.MCTS,
                    MctsNumSimulations = 32,
                    MctsMaxSearchDepth = 8,
                    MctsRolloutDepth = 4,
                    MctsExplorationConstant = 1.414f,
                    MctsGamma = 0.99f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "ppo_continuous_target_smoke",
                Name = "PPO / Continuous Target / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.ContinuousTarget1D,
                Algorithm = RLAlgorithmKind.PPO,
                IsImplemented = true,
                Notes = "Continuous-action PPO smoke on a tiny 1D control task.",
                CreateEnvironment = static () => new ContinuousTarget1DBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.PPO,
                    RolloutLength = 32,
                    EpochsPerUpdate = 2,
                    PpoMiniBatchSize = 16,
                    LearningRate = 0.001f,
                    Gamma = 0.99f,
                    GaeLambda = 0.95f,
                    ClipEpsilon = 0.2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = true,
                    ValueClipEpsilon = 0.2f,
                    EntropyCoefficient = 0.01f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "a2c_continuous_target_smoke",
                Name = "A2C / Continuous Target / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.ContinuousTarget1D,
                Algorithm = RLAlgorithmKind.A2C,
                IsImplemented = true,
                Notes = "Continuous-action A2C smoke to validate the same task with a lighter update rule.",
                CreateEnvironment = static () => new ContinuousTarget1DBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.A2C,
                    RolloutLength = 32,
                    EpochsPerUpdate = 1,
                    PpoMiniBatchSize = 32,
                    LearningRate = 0.001f,
                    Gamma = 0.99f,
                    GaeLambda = 1.0f,
                    ClipEpsilon = float.MaxValue / 2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = false,
                    ValueClipEpsilon = 0f,
                    EntropyCoefficient = 0.01f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "sac_continuous_target_smoke",
                Name = "SAC / Continuous Target / Smoke",
                Suite = AlgorithmBenchSuite.Smoke,
                Task = AlgorithmBenchTaskKind.ContinuousTarget1D,
                Algorithm = RLAlgorithmKind.SAC,
                StepBudget = 384,
                EpisodeBudget = 24,
                IsImplemented = true,
                Notes = "Exercises replay, actor-critic updates, and alpha tracking on a tiny continuous task.",
                CreateEnvironment = static () => new ContinuousTarget1DBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.SAC,
                    LearningRate = 0.0005f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 1.0f,
                    ReplayBufferCapacity = 4096,
                    SacBatchSize = 32,
                    SacWarmupSteps = 32,
                    SacTau = 0.01f,
                    SacInitAlpha = 0.2f,
                    SacAutoTuneAlpha = true,
                    SacUpdateEverySteps = 1,
                    SacUpdatesPerStep = 1,
                    SacTargetEntropyFraction = 0.5f,
                    SacContinuousTargetEntropyScale = 1.0f,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "ppo_discrete_bandit_learn",
                Name = "PPO / Discrete Bandit / Learn",
                Suite = AlgorithmBenchSuite.Learn,
                Task = AlgorithmBenchTaskKind.DiscreteBandit,
                Algorithm = RLAlgorithmKind.PPO,
                IsImplemented = false,
                Notes = "Planned learning-regression case. Needs seeded trainer RNG and tuned thresholds before it should gate changes.",
                Thresholds = new AlgorithmBenchThresholds { MinMeanReward = 0.85f, MaxStepsToThreshold = 512 },
                CreateEnvironment = static () => new DiscreteBanditBenchEnvironment(),
                CreateTrainerConfig = static () => new RLTrainerConfig(),
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "ppo_synth_small_perf",
                Name = "PPO / Synthetic Discrete / Small Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.PPO,
                IsImplemented = true,
                Notes = "L1 batched throughput run with a small network and moderate fake env cost.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(64, 6, 24, 12),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.PPO,
                    RolloutLength = 128,
                    EpochsPerUpdate = 2,
                    PpoMiniBatchSize = 64,
                    LearningRate = 0.0007f,
                    Gamma = 0.99f,
                    GaeLambda = 0.95f,
                    ClipEpsilon = 0.2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = true,
                    ValueClipEpsilon = 0.2f,
                    EntropyCoefficient = 0.01f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 32,
                    WarmupTicks = 32,
                    MeasureTicks = 192,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "ppo_synth_large_perf",
                Name = "PPO / Synthetic Discrete / Large Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.PPO,
                IsImplemented = true,
                Notes = "L1 batched throughput run with a large network and heavier fake env cost.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(512, 12, 24, 40),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.PPO,
                    RolloutLength = 128,
                    EpochsPerUpdate = 2,
                    PpoMiniBatchSize = 64,
                    LearningRate = 0.0005f,
                    Gamma = 0.99f,
                    GaeLambda = 0.95f,
                    ClipEpsilon = 0.2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = true,
                    ValueClipEpsilon = 0.2f,
                    EntropyCoefficient = 0.01f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 64,
                    WarmupTicks = 48,
                    MeasureTicks = 224,
                },
                CreateNetworkGraph = CreateLargeGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "a2c_synth_small_perf",
                Name = "A2C / Synthetic Discrete / Small Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.A2C,
                IsImplemented = true,
                Notes = "Single-pass actor-critic throughput on the same synthetic discrete env.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(64, 6, 24, 12),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.A2C,
                    RolloutLength = 128,
                    EpochsPerUpdate = 1,
                    PpoMiniBatchSize = 128,
                    LearningRate = 0.0007f,
                    Gamma = 0.99f,
                    GaeLambda = 1.0f,
                    ClipEpsilon = float.MaxValue / 2f,
                    MaxGradientNorm = 0.5f,
                    ValueLossCoefficient = 0.5f,
                    UseValueClipping = false,
                    ValueClipEpsilon = 0f,
                    EntropyCoefficient = 0.01f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 32,
                    WarmupTicks = 32,
                    MeasureTicks = 192,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "dqn_synth_small_perf",
                Name = "DQN / Synthetic Discrete / Small Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.DQN,
                IsImplemented = true,
                Notes = "Replay-buffer throughput on a moderate synthetic discrete env.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(64, 8, 24, 14),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.DQN,
                    LearningRate = 0.0007f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 5f,
                    ReplayBufferCapacity = 8192,
                    DqnBatchSize = 64,
                    DqnWarmupSteps = 128,
                    DqnEpsilonStart = 1.0f,
                    DqnEpsilonEnd = 0.05f,
                    DqnEpsilonDecaySteps = 2000,
                    DqnTargetUpdateInterval = 128,
                    DqnUseDouble = true,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 64,
                    WarmupTicks = 48,
                    MeasureTicks = 256,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "dqn_synth_large_perf",
                Name = "DQN / Synthetic Discrete / Large Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.DQN,
                IsImplemented = true,
                Notes = "Heavier replay-buffer throughput case with larger observation vectors and larger MLP.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(512, 12, 24, 40),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.DQN,
                    LearningRate = 0.0005f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 5f,
                    ReplayBufferCapacity = 16384,
                    DqnBatchSize = 128,
                    DqnWarmupSteps = 256,
                    DqnEpsilonStart = 1.0f,
                    DqnEpsilonEnd = 0.05f,
                    DqnEpsilonDecaySteps = 4000,
                    DqnTargetUpdateInterval = 256,
                    DqnUseDouble = true,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 128,
                    WarmupTicks = 64,
                    MeasureTicks = 256,
                },
                CreateNetworkGraph = CreateLargeGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "sac_synth_small_perf",
                Name = "SAC / Synthetic Continuous / Small Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.ContinuousTarget1D,
                Algorithm = RLAlgorithmKind.SAC,
                IsImplemented = true,
                Notes = "Continuous-control actor-critic throughput on a moderate synthetic vector env.",
                CreateEnvironment = static () => new SyntheticContinuousVectorBenchEnvironment(96, 3, 24, 16),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.SAC,
                    LearningRate = 0.0005f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 1f,
                    ReplayBufferCapacity = 8192,
                    SacBatchSize = 64,
                    SacWarmupSteps = 128,
                    SacTau = 0.01f,
                    SacInitAlpha = 0.2f,
                    SacAutoTuneAlpha = true,
                    SacUpdateEverySteps = 1,
                    SacUpdatesPerStep = 1,
                    SacTargetEntropyFraction = 0.5f,
                    SacContinuousTargetEntropyScale = 1.0f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 64,
                    WarmupTicks = 48,
                    MeasureTicks = 224,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "sac_synth_large_perf",
                Name = "SAC / Synthetic Continuous / Large Net / Performance",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.ContinuousTarget1D,
                Algorithm = RLAlgorithmKind.SAC,
                IsImplemented = true,
                Notes = "Large-network continuous-control throughput case with heavier fake env cost.",
                CreateEnvironment = static () => new SyntheticContinuousVectorBenchEnvironment(512, 6, 24, 40),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.SAC,
                    LearningRate = 0.0003f,
                    Gamma = 0.99f,
                    MaxGradientNorm = 1f,
                    ReplayBufferCapacity = 16384,
                    SacBatchSize = 128,
                    SacWarmupSteps = 256,
                    SacTau = 0.01f,
                    SacInitAlpha = 0.2f,
                    SacAutoTuneAlpha = true,
                    SacUpdateEverySteps = 1,
                    SacUpdatesPerStep = 1,
                    SacTargetEntropyFraction = 0.5f,
                    SacContinuousTargetEntropyScale = 1.0f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 96,
                    WarmupTicks = 64,
                    MeasureTicks = 224,
                },
                CreateNetworkGraph = CreateLargeGraph,
            },
            new AlgorithmBenchCase
            {
                Id = "mcts_synth_perf",
                Name = "MCTS / Synthetic Discrete / Planning Throughput",
                Suite = AlgorithmBenchSuite.Performance,
                Task = AlgorithmBenchTaskKind.DiscreteLineWorld,
                Algorithm = RLAlgorithmKind.MCTS,
                IsImplemented = true,
                Notes = "Planning-heavy synthetic throughput case. Measures decision latency rather than training updates.",
                CreateEnvironment = static () => new SyntheticDiscreteVectorBenchEnvironment(96, 8, 16, 24),
                CreateTrainerConfig = static () => new RLTrainerConfig
                {
                    Algorithm = RLAlgorithmKind.MCTS,
                    MctsNumSimulations = 96,
                    MctsMaxSearchDepth = 10,
                    MctsRolloutDepth = 6,
                    MctsExplorationConstant = 1.414f,
                    MctsGamma = 0.99f,
                },
                Performance = new AlgorithmBenchPerformanceConfig
                {
                    ParallelActors = 16,
                    WarmupTicks = 8,
                    MeasureTicks = 64,
                },
                CreateNetworkGraph = CreateCompactGraph,
            },
        };
    }

    private static RLNetworkGraph CreateCompactGraph()
    {
        return new RLNetworkGraph
        {
            UseNativeLayers = false,
            Optimizer = RLOptimizerKind.Adam,
            TrunkLayers = new Array<Resource>
            {
                new RLDenseLayerDef { Size = 32, Activation = RLActivationKind.Tanh },
                new RLDenseLayerDef { Size = 32, Activation = RLActivationKind.Tanh },
            },
        };
    }

    private static RLNetworkGraph CreateLargeGraph()
    {
        return new RLNetworkGraph
        {
            UseNativeLayers = false,
            Optimizer = RLOptimizerKind.Adam,
            TrunkLayers = new Array<Resource>
            {
                new RLDenseLayerDef { Size = 256, Activation = RLActivationKind.Tanh },
                new RLDenseLayerDef { Size = 256, Activation = RLActivationKind.Tanh },
                new RLDenseLayerDef { Size = 128, Activation = RLActivationKind.Tanh },
            },
        };
    }
}
