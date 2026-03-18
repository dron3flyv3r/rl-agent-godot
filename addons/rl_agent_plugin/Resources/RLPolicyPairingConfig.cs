using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLPolicyPairingConfig : Resource
{
    private Resource? _groupA;
    private Resource? _groupB;

    [Export] public string PairingId { get; set; } = string.Empty;

    [ExportGroup("Groups")]
    [Export(PropertyHint.ResourceType, nameof(RLPolicyGroupConfig))]
    public Resource? GroupA
    {
        get => _groupA;
        set => _groupA = value;
    }

    [Export(PropertyHint.ResourceType, nameof(RLPolicyGroupConfig))]
    public Resource? GroupB
    {
        get => _groupB;
        set => _groupB = value;
    }

    [ExportGroup("Training")]
    [Export] public bool TrainGroupA { get; set; } = true;
    [Export] public bool TrainGroupB { get; set; } = true;
    [Export(PropertyHint.Range, "0,1,0.01")] public float HistoricalOpponentRate { get; set; } = 0.5f;
    [Export(PropertyHint.Range, "1,100000,1,or_greater")] public int FrozenCheckpointInterval { get; set; } = 10;

    [ExportGroup("PFSP")]
    [Export] public int   MaxPoolSize  { get; set; } = 20;
    [Export] public bool  PfspEnabled  { get; set; } = true;
    [Export] public float PfspAlpha    { get; set; } = 4.0f;
    [Export] public float WinThreshold { get; set; } = 0.0f;

    public RLPolicyGroupConfig? ResolvedGroupA => _groupA as RLPolicyGroupConfig;
    public RLPolicyGroupConfig? ResolvedGroupB => _groupB as RLPolicyGroupConfig;
}
