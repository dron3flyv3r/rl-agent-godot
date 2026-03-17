using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLPolicyGroupConfig : Resource
{
    private Resource? _networkGraph;

    [Export] public string AgentId { get; set; } = string.Empty;
    [Export] public int MaxEpisodeSteps { get; set; } = 0;
    [Export(PropertyHint.File, "*.rlmodel")] public string InferenceModelPath { get; set; } = string.Empty;

    [ExportGroup("Network")]
    [Export(PropertyHint.ResourceType, nameof(RLNetworkGraph))]
    public Resource? NetworkGraph
    {
        get => _networkGraph;
        set => _networkGraph = value;
    }

    public RLNetworkGraph? ResolvedNetworkGraph => _networkGraph as RLNetworkGraph;
}
