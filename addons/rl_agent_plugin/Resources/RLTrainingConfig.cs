using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLTrainingConfig : Resource
{
    [ExportGroup("Algorithm")]
    [Export] public RLAlgorithmKind Algorithm { get; set; } = RLAlgorithmKind.PPO;

    [ExportGroup("Network")]
    [Export] public int[] HiddenLayerSizes { get; set; } = new[] { 64, 64 };
    [Export] public RLActivationKind Activation { get; set; } = RLActivationKind.Tanh;
    [Export] public bool SharedTrunk { get; set; } = true;
    [Export] public RLOptimizerKind Optimizer { get; set; } = RLOptimizerKind.Adam;

    [ExportGroup("PPO")]
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

    [ExportGroup("SAC")]
    [Export] public int ReplayBufferCapacity { get; set; } = 100_000;
    [Export] public int SacBatchSize { get; set; } = 256;
    [Export] public int SacWarmupSteps { get; set; } = 1_000;
    [Export] public float SacTau { get; set; } = 0.005f;
    [Export] public float SacInitAlpha { get; set; } = 0.2f;
    [Export] public bool SacAutoTuneAlpha { get; set; } = true;
    [Export] public int SacUpdateEverySteps { get; set; } = 1;

    public RLTrainerConfig ToTrainerConfig()
    {
        return new RLTrainerConfig
        {
            Algorithm = Algorithm,
            RolloutLength = RolloutLength,
            EpochsPerUpdate = EpochsPerUpdate,
            LearningRate = LearningRate,
            Gamma = Gamma,
            GaeLambda = GaeLambda,
            ClipEpsilon = ClipEpsilon,
            ValueLossCoefficient = ValueLossCoefficient,
            EntropyCoefficient = EntropyCoefficient,
            MaxEpisodeSteps = MaxEpisodeSteps,
            StatusWriteIntervalSteps = StatusWriteIntervalSteps,
            CheckpointIntervalUpdates = CheckpointIntervalUpdates,
            ReplayBufferCapacity = ReplayBufferCapacity,
            SacBatchSize = SacBatchSize,
            SacWarmupSteps = SacWarmupSteps,
            SacTau = SacTau,
            SacInitAlpha = SacInitAlpha,
            SacAutoTuneAlpha = SacAutoTuneAlpha,
            SacUpdateEverySteps = SacUpdateEverySteps,
        };
    }

    public RLNetworkConfig ToNetworkConfig()
    {
        return new RLNetworkConfig
        {
            HiddenLayerSizes = HiddenLayerSizes,
            Activation = Activation,
            SharedTrunk = SharedTrunk,
            Optimizer = Optimizer,
        };
    }
}
