using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// PPO (Proximal Policy Optimization) hyperparameters.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Algorithm"/>.
/// </summary>
[GlobalClass]
[Tool]
public partial class RLPPOConfig : RLAlgorithmConfig
{
    [Export] public int   RolloutLength         { get; set; } = 256;
    [Export] public int   EpochsPerUpdate        { get; set; } = 4;
    [Export] public int   MiniBatchSize          { get; set; } = 64;
    [Export] public float LearningRate           { get; set; } = 0.0005f;
    [Export] public float Gamma                  { get; set; } = 0.99f;
    [Export] public float GaeLambda              { get; set; } = 0.95f;
    [Export] public float ClipEpsilon            { get; set; } = 0.2f;
    [Export] public float MaxGradientNorm        { get; set; } = 0.5f;
    [Export] public float ValueLossCoefficient   { get; set; } = 0.5f;
    [Export] public bool  UseValueClipping       { get; set; } = true;
    [Export] public float ValueClipEpsilon       { get; set; } = 0.2f;
    [Export] public float EntropyCoefficient     { get; set; } = 0.01f;

    public override RLAlgorithmKind AlgorithmKind => RLAlgorithmKind.PPO;

    internal override void ApplyTo(RLTrainerConfig config)
    {
        config.Algorithm               = RLAlgorithmKind.PPO;
        config.RolloutLength           = RolloutLength;
        config.EpochsPerUpdate         = EpochsPerUpdate;
        config.PpoMiniBatchSize        = MiniBatchSize;
        config.LearningRate            = LearningRate;
        config.Gamma                   = Gamma;
        config.GaeLambda               = GaeLambda;
        config.ClipEpsilon             = ClipEpsilon;
        config.MaxGradientNorm         = MaxGradientNorm;
        config.ValueLossCoefficient    = ValueLossCoefficient;
        config.UseValueClipping        = UseValueClipping;
        config.ValueClipEpsilon        = ValueClipEpsilon;
        config.EntropyCoefficient      = EntropyCoefficient;
        config.StatusWriteIntervalSteps  = StatusWriteIntervalSteps;
        config.CheckpointIntervalUpdates = CheckpointIntervalUpdates;
    }
}
