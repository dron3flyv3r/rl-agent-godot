using Godot;

namespace RlAgentPlugin.Runtime;

public enum RLCurriculumMode
{
    StepBased = 0,
    SuccessRate = 1,
}

[GlobalClass]
[Tool]
public partial class RLCurriculumConfig : Resource
{
    [Export] public RLCurriculumMode Mode { get; set; } = RLCurriculumMode.StepBased;
    [Export(PropertyHint.Range, "0,10000000,1,or_greater")] public long MaxSteps { get; set; } = 0;
    [Export(PropertyHint.Range, "1,10000,1,or_greater")] public int SuccessWindowEpisodes { get; set; } = 25;
    [Export] public float SuccessRewardThreshold { get; set; } = 1.0f;
    [Export(PropertyHint.Range, "0,1,0.01")] public float PromoteThreshold { get; set; } = 0.8f;
    [Export(PropertyHint.Range, "0,1,0.01")] public float DemoteThreshold { get; set; } = 0.2f;
    [Export(PropertyHint.Range, "0,1,0.01")] public float ProgressStepUp { get; set; } = 0.1f;
    [Export(PropertyHint.Range, "0,1,0.01")] public float ProgressStepDown { get; set; } = 0.1f;
    [Export] public bool RequireFullWindow { get; set; } = true;
    [Export(PropertyHint.Range, "0,1,0.01")] public float DebugProgress { get; set; } = 0f;
}
