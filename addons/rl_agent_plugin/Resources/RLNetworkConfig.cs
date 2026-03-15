using Godot;

namespace RlAgentPlugin.Runtime;

public enum RLActivationKind
{
    Tanh = 0,
    Relu = 1,
}

public enum RLOptimizerKind
{
    Adam = 0,
    Sgd = 1,
}

[GlobalClass]
[Tool]
public partial class RLNetworkConfig : Resource
{
    [Export] public int[] HiddenLayerSizes { get; set; } = new[] { 64, 64 };
    [Export] public RLActivationKind Activation { get; set; } = RLActivationKind.Tanh;
    [Export] public bool SharedTrunk { get; set; } = true;
    [Export] public RLOptimizerKind Optimizer { get; set; } = RLOptimizerKind.Adam;
}
