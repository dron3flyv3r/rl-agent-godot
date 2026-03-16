using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLPolicyGroupConfig : Resource
{
    [Export] public string GroupId { get; set; } = string.Empty;
    [Export(PropertyHint.File, "*.json,*.rlmodel")] public string InferenceCheckpointPath { get; set; } = string.Empty;
    [Export] public bool SelfPlay { get; set; }
    [Export(PropertyHint.Range, "0,1,0.01")] public float HistoricalOpponentRate { get; set; } = 0.5f;
    [Export(PropertyHint.Range, "1,100000,1,or_greater")] public int FrozenCheckpointInterval { get; set; } = 10;
}
