using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLTrainerConfig : Resource
{
    // ── Algorithm selection ─────────────────────────────────────────────────
    [Export] public RLAlgorithmKind Algorithm { get; set; } = RLAlgorithmKind.PPO;

    // ── PPO hyperparameters ─────────────────────────────────────────────────
    [Export] public int RolloutLength { get; set; } = 256;
    [Export] public int EpochsPerUpdate { get; set; } = 4;
    [Export] public float LearningRate { get; set; } = 0.0005f;
    [Export] public float Gamma { get; set; } = 0.99f;
    [Export] public float GaeLambda { get; set; } = 0.95f;
    [Export] public float ClipEpsilon { get; set; } = 0.2f;
    [Export] public float ValueLossCoefficient { get; set; } = 0.5f;
    [Export] public float EntropyCoefficient { get; set; } = 0.01f;
    [Export] public int MaxEpisodeSteps { get; set; } = 1024;
    [Export] public int StatusWriteIntervalSteps { get; set; } = 32;
    [Export] public int CheckpointIntervalUpdates { get; set; } = 10;

    // ── SAC hyperparameters (ignored by PPO) ───────────────────────────────
    [Export] public int ReplayBufferCapacity { get; set; } = 100_000;
    [Export] public int SacBatchSize { get; set; } = 256;
    [Export] public int SacWarmupSteps { get; set; } = 1_000;
    [Export] public float SacTau { get; set; } = 0.005f;
    [Export] public float SacInitAlpha { get; set; } = 0.2f;
    [Export] public bool SacAutoTuneAlpha { get; set; } = true;
    [Export] public int SacUpdateEverySteps { get; set; } = 1;
}
