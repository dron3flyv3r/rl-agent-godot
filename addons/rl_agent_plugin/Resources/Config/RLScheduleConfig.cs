using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Optional per-hyperparameter schedules.
/// Create this as a .tres resource and assign it to <see cref="RLTrainingConfig.Schedules"/>.
///
/// When a schedule is assigned it overrides the corresponding flat value on
/// each gradient update. Leave a slot null to keep the flat value constant.
///
/// Built-in types: RLConstantSchedule, RLLinearSchedule,
///                 RLExponentialSchedule, RLCosineSchedule.
/// Custom: subclass RLHyperparamSchedule and add [GlobalClass].
/// </summary>
[GlobalClass]
[Tool]
public partial class RLScheduleConfig : Resource
{
    [Export] public RLHyperparamSchedule? LearningRate      { get; set; }
    [Export] public RLHyperparamSchedule? EntropyCoefficient { get; set; }
    [Export] public RLHyperparamSchedule? ClipEpsilon        { get; set; }
    [Export] public RLHyperparamSchedule? SacAlpha           { get; set; }

    internal void ApplyTo(RLTrainerConfig config, ScheduleContext ctx)
    {
        if (LearningRate       is not null) config.LearningRate      = LearningRate.Evaluate(ctx);
        if (EntropyCoefficient is not null) config.EntropyCoefficient = EntropyCoefficient.Evaluate(ctx);
        if (ClipEpsilon        is not null) config.ClipEpsilon        = ClipEpsilon.Evaluate(ctx);
        if (SacAlpha           is not null) config.SacInitAlpha       = SacAlpha.Evaluate(ctx);
    }
}
