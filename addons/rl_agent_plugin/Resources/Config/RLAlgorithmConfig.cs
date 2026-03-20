using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Base class for algorithm configuration resources.
/// Assign an <see cref="RLPPOConfig"/> or <see cref="RLSACConfig"/> asset to
/// <see cref="RLTrainingConfig.Algorithm"/> in the Inspector.
///
/// To use a custom trainer: subclass this, add [GlobalClass], set
/// <see cref="RLTrainerConfig.Algorithm"/> to <see cref="RLAlgorithmKind.Custom"/>
/// and fill <see cref="RLTrainerConfig.CustomTrainerId"/> inside <see cref="ApplyTo"/>.
/// </summary>
[GlobalClass]
[Tool]
public abstract partial class RLAlgorithmConfig : Resource
{
    [Export] public int StatusWriteIntervalSteps  { get; set; } = 32;
    [Export] public int CheckpointIntervalUpdates { get; set; } = 10;

    /// <summary>The algorithm this config represents. Implemented by each concrete subclass.</summary>
    public virtual RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.Custom;

    /// <summary>Writes all settings from this config into <paramref name="config"/>.</summary>
    internal abstract void ApplyTo(RLTrainerConfig config);
}
