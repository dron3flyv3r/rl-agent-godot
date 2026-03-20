using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLSelfPlayConfig : Resource
{
    [Export] public Godot.Collections.Array<RLPolicyPairingConfig> Pairings { get; set; } = new();
}
