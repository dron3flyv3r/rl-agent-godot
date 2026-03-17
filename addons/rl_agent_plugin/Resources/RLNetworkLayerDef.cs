using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLNetworkLayerDef : Resource
{
    [Export] public int Size { get; set; } = 64;
    [Export] public RLActivationKind Activation { get; set; } = RLActivationKind.Tanh;
}
