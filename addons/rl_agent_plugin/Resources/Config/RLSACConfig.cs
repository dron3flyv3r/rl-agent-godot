using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// SAC (Soft Actor-Critic) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLSACConfig : RLAlgorithmConfig
{
    [Export] public float LearningRate          { get; set; } = 0.0003f;
    [Export] public float Gamma                 { get; set; } = 0.99f;
    [Export] public float MaxGradientNorm       { get; set; } = 0.5f;
    [Export] public int   ReplayBufferCapacity  { get; set; } = 100_000;
    [Export] public int   BatchSize             { get; set; } = 256;
    [Export] public int   WarmupSteps           { get; set; } = 1_000;
    [Export] public float Tau                   { get; set; } = 0.005f;
    [Export] public float InitAlpha             { get; set; } = 0.2f;
    [Export] public bool  AutoTuneAlpha         { get; set; } = true;
    [Export] public int   UpdateEverySteps      { get; set; } = 1;
    /// <summary>
    /// Fraction of maximum entropy used as the discrete-action target entropy.
    /// 1.0 = fully random (uniform policy); 0.5 = half of maximum entropy.
    /// Lower values make the policy converge to more deterministic behaviour.
    /// Only used when the action space is discrete (ignored for continuous).
    /// </summary>
    [Export] public float TargetEntropyFraction { get; set; } = 0.5f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.SAC;

    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm               = RLAlgorithmKind.SAC;
        config.LearningRate            = LearningRate;
        config.Gamma                   = Gamma;
        config.MaxGradientNorm         = MaxGradientNorm;
        config.ReplayBufferCapacity    = ReplayBufferCapacity;
        config.SacBatchSize            = BatchSize;
        config.SacWarmupSteps          = WarmupSteps;
        config.SacTau                  = Tau;
        config.SacInitAlpha               = InitAlpha;
        config.SacAutoTuneAlpha           = AutoTuneAlpha;
        config.SacUpdateEverySteps        = UpdateEverySteps;
        config.SacTargetEntropyFraction   = TargetEntropyFraction;
        config.StatusWriteIntervalSteps   = StatusWriteIntervalSteps;
        config.CheckpointIntervalUpdates = CheckpointIntervalUpdates;
    }
}
