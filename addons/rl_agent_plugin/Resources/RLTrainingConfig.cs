using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLTrainingConfig : Resource
{
    [ExportGroup("Algorithm")]
    [Export] public RLAlgorithmKind Algorithm { get; set; } = RLAlgorithmKind.PPO;
    /// <summary>
    /// Used only when <see cref="Algorithm"/> is <see cref="RLAlgorithmKind.Custom"/>.
    /// Must match the key passed to <see cref="TrainerFactory.Register"/>.
    /// </summary>
    [Export] public string CustomTrainerId { get; set; } = string.Empty;

    [ExportGroup("PPO")]
    [Export] public int RolloutLength { get; set; } = 256;
    [Export] public int EpochsPerUpdate { get; set; } = 4;
    [Export] public int PpoMiniBatchSize { get; set; } = 64;
    [Export] public float LearningRate { get; set; } = 0.0005f;
    [Export] public float Gamma { get; set; } = 0.99f;
    [Export] public float GaeLambda { get; set; } = 0.95f;
    [Export] public float ClipEpsilon { get; set; } = 0.2f;
    [Export] public float MaxGradientNorm { get; set; } = 0.5f;
    [Export] public float ValueLossCoefficient { get; set; } = 0.5f;
    [Export] public bool UseValueClipping { get; set; } = true;
    [Export] public float ValueClipEpsilon { get; set; } = 0.2f;
    [Export] public float EntropyCoefficient { get; set; } = 0.01f;
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

    /// <summary>
    /// Optional per-hyperparameter schedules. When a schedule is assigned it
    /// overrides the corresponding flat value at each gradient update. Leave null
    /// to keep the flat value constant throughout training.
    ///
    /// Built-in types: RLConstantSchedule, RLLinearSchedule,
    ///                 RLExponentialSchedule, RLCosineSchedule.
    /// Custom: subclass RLHyperparamSchedule and add [GlobalClass].
    /// </summary>
    [ExportGroup("Schedules")]
    [Export] public RLHyperparamSchedule? LearningRateSchedule      { get; set; }
    [Export] public RLHyperparamSchedule? EntropyCoefficientSchedule { get; set; }
    [Export] public RLHyperparamSchedule? ClipEpsilonSchedule        { get; set; }
    [Export] public RLHyperparamSchedule? SacAlphaSchedule           { get; set; }

    /// <summary>
    /// Evaluates all non-null schedules and writes their results into <paramref name="config"/>.
    /// Called by TrainingBootstrap once per gradient update, before TryUpdate().
    /// </summary>
    internal void ApplySchedules(RLTrainerConfig config, ScheduleContext ctx)
    {
        if (LearningRateSchedule      is not null) config.LearningRate        = LearningRateSchedule.Evaluate(ctx);
        if (EntropyCoefficientSchedule is not null) config.EntropyCoefficient  = EntropyCoefficientSchedule.Evaluate(ctx);
        if (ClipEpsilonSchedule        is not null) config.ClipEpsilon         = ClipEpsilonSchedule.Evaluate(ctx);
        if (SacAlphaSchedule           is not null) config.SacInitAlpha        = SacAlphaSchedule.Evaluate(ctx);
    }

    public RLTrainerConfig ToTrainerConfig()
    {
        return new RLTrainerConfig
        {
            Algorithm = Algorithm,
            CustomTrainerId = CustomTrainerId,
            RolloutLength = RolloutLength,
            EpochsPerUpdate = EpochsPerUpdate,
            PpoMiniBatchSize = PpoMiniBatchSize,
            LearningRate = LearningRate,
            Gamma = Gamma,
            GaeLambda = GaeLambda,
            ClipEpsilon = ClipEpsilon,
            MaxGradientNorm = MaxGradientNorm,
            ValueLossCoefficient = ValueLossCoefficient,
            UseValueClipping = UseValueClipping,
            ValueClipEpsilon = ValueClipEpsilon,
            EntropyCoefficient = EntropyCoefficient,
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

}
