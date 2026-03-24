using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLSelfPlayConfig : Resource
{
    /// <summary>
    /// List of policy-group matchups used to configure self-play training.
    /// </summary>
    [Export] public Godot.Collections.Array<RLPolicyPairingConfig> Pairings { get; set; } = new();
}
