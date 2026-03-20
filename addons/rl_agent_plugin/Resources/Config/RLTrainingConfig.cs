using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLTrainingConfig : Resource
{
    private const string AlgorithmTypes = $"{nameof(RLPPOConfig)},{nameof(RLSACConfig)}";

    [Export(PropertyHint.ResourceType, AlgorithmTypes)]
    public RLAlgorithmConfig? Algorithm { get; set; } = new RLPPOConfig();

    [Export]
    public RLScheduleConfig? Schedules { get; set; }

    /// <summary>
    /// Evaluates all non-null schedules and writes their results into <paramref name="config"/>.
    /// Called by TrainingBootstrap once per gradient update, before TryUpdate().
    /// </summary>
    internal void ApplySchedules(RLTrainerConfig config, ScheduleContext ctx)
    {
        Schedules?.ApplyTo(config, ctx);
    }

    public RLTrainerConfig? ToTrainerConfig()
    {
        if (Algorithm is null) return null;
        var config = new RLTrainerConfig();
        Algorithm.ApplyTo(config);
        return config;
    }
}
