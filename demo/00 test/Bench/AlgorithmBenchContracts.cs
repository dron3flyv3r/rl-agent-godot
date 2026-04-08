using System;
using System.Collections.Generic;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Demo.Benchmarks;

public enum AlgorithmBenchRunMode
{
    Catalog = 0,
    Smoke = 1,
    Performance = 2,
    AllImplemented = 3,
}

public enum AlgorithmBenchSuite
{
    Smoke = 0,
    Learn = 1,
    Performance = 2,
}

public enum AlgorithmBenchTaskKind
{
    DiscreteBandit = 0,
    DiscreteLineWorld = 1,
    ContinuousTarget1D = 2,
}

public readonly record struct AlgorithmBenchAction(int DiscreteAction, float[] ContinuousActions);

public readonly record struct AlgorithmBenchStep(float[] Observation, float Reward, bool Done);

public sealed class AlgorithmBenchThresholds
{
    public float? MinMeanReward { get; init; }
    public long? MaxStepsToThreshold { get; init; }
    public double? MaxDecisionMillisecondsP95 { get; init; }
}

public sealed class AlgorithmBenchPerformanceConfig
{
    public int ParallelActors { get; init; } = 16;
    public int WarmupTicks { get; init; } = 32;
    public int MeasureTicks { get; init; } = 256;
}

public sealed class AlgorithmBenchCase
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public AlgorithmBenchSuite Suite { get; init; }
    public AlgorithmBenchTaskKind Task { get; init; }
    public RLAlgorithmKind Algorithm { get; init; }
    public int Seed { get; init; } = 1;
    public int StepBudget { get; init; } = 256;
    public int EpisodeBudget { get; init; } = 16;
    public bool IsImplemented { get; init; }
    public string Notes { get; init; } = string.Empty;
    public AlgorithmBenchThresholds Thresholds { get; init; } = new();
    public AlgorithmBenchPerformanceConfig Performance { get; init; } = new();
    public Func<IAlgorithmBenchEnvironment> CreateEnvironment { get; init; } = default!;
    public Func<RLTrainerConfig> CreateTrainerConfig { get; init; } = default!;
    public Func<RLNetworkGraph> CreateNetworkGraph { get; init; } = default!;
}

public sealed class AlgorithmBenchResult
{
    public string CaseId { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public RLAlgorithmKind Algorithm { get; init; }
    public AlgorithmBenchSuite Suite { get; init; }
    public bool Passed { get; init; }
    public int Episodes { get; init; }
    public int Steps { get; init; }
    public int Updates { get; init; }
    public float MeanEpisodeReward { get; init; }
    public double ElapsedMilliseconds { get; init; }
    public double EnvStepsPerSecond { get; init; }
    public double DecisionsPerSecond { get; init; }
    public double UpdatesPerSecond { get; init; }
    public double DecisionMillisecondsP95 { get; init; }
    public double UpdateMillisecondsP95 { get; init; }
    public string Detail { get; init; } = string.Empty;
}

public interface IAlgorithmBenchEnvironment
{
    string Name { get; }
    int ObservationSize { get; }
    int DiscreteActionCount { get; }
    int ContinuousActionDimensions { get; }
    float[] Reset(int seed);
    AlgorithmBenchStep Step(AlgorithmBenchAction action);
}

public interface IPlanningBenchEnvironment : IAlgorithmBenchEnvironment, IEnvironmentModel
{
}

public static class AlgorithmBenchFormatting
{
    public static string JoinLabels(IEnumerable<RLAlgorithmKind> algorithms)
        => string.Join(", ", algorithms);
}
