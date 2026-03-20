using System;

namespace RlAgentPlugin.Runtime;

public enum RLAlgorithmKind
{
    PPO = 0,
    SAC = 1,
    /// <summary>
    /// Use a custom trainer registered via <see cref="TrainerFactory.Register"/>.
    /// Set <see cref="PolicyGroupConfig.CustomTrainerId"/> to the registered key.
    /// </summary>
    Custom = 99,
}

public sealed class PolicyDecision
{
    /// <summary>Sampled discrete action index, or -1 if continuous-only.</summary>
    public int DiscreteAction { get; init; } = -1;
    /// <summary>Sampled continuous action vector (empty for discrete-only).</summary>
    public float[] ContinuousActions { get; init; } = Array.Empty<float>();
    public float LogProbability { get; init; }
    public float Value { get; init; }
    public float Entropy { get; init; }
}

public sealed class Transition
{
    public float[] Observation { get; init; } = Array.Empty<float>();
    /// <summary>Taken discrete action index, or -1 if continuous.</summary>
    public int DiscreteAction { get; init; } = -1;
    public float[] ContinuousActions { get; init; } = Array.Empty<float>();
    public float Reward { get; init; }
    public bool Done { get; init; }
    /// <summary>Next observation (used by SAC for target Q computation).</summary>
    public float[] NextObservation { get; init; } = Array.Empty<float>();
    /// <summary>Log probability of the taken action (used by PPO).</summary>
    public float OldLogProbability { get; init; }
    /// <summary>Value estimate at this state (used by PPO for GAE).</summary>
    public float Value { get; init; }
    /// <summary>Value estimate at next state (used by PPO for GAE).</summary>
    public float NextValue { get; init; }
}

public sealed class TrainerUpdateStats
{
    public float PolicyLoss { get; init; }
    public float ValueLoss { get; init; }
    public float Entropy { get; init; }
    public float ClipFraction { get; init; }
    public RLCheckpoint Checkpoint { get; init; } = new();
}

public sealed class PolicyGroupConfig
{
    public string GroupId { get; init; } = string.Empty;
    public string RunId { get; init; } = string.Empty;
    public RLAlgorithmKind Algorithm { get; init; } = RLAlgorithmKind.PPO;
    /// <summary>Key passed to <see cref="TrainerFactory.Register"/> when Algorithm is Custom.</summary>
    public string CustomTrainerId { get; init; } = string.Empty;
    public RLPolicyGroupConfig? SharedPolicy { get; init; }
    public RLTrainerConfig TrainerConfig { get; init; } = new();
    public RLNetworkGraph NetworkGraph { get; init; } = new();
    public RLActionDefinition[] ActionDefinitions { get; init; } = Array.Empty<RLActionDefinition>();
    public int ObservationSize { get; init; }
    public int DiscreteActionCount { get; init; }
    public int ContinuousActionDimensions { get; init; }
    public string CheckpointPath { get; init; } = string.Empty;
    public string MetricsPath { get; init; } = string.Empty;
}

public interface ITrainer
{
    PolicyDecision SampleAction(float[] observation);
    PolicyDecision[] SampleActions(VectorBatch observations);
    /// <summary>Returns a value estimate (PPO: value head; SAC: returns 0).</summary>
    float EstimateValue(float[] observation);
    float[] EstimateValues(VectorBatch observations);
    void RecordTransition(Transition transition);
    TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount);
    RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount);
}
