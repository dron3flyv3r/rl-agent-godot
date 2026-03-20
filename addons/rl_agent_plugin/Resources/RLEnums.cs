namespace RlAgentPlugin.Runtime;

public enum RLActivationKind
{
    Tanh = 0,
    Relu = 1,
}

public enum RLOptimizerKind
{
    Adam = 0,
    Sgd  = 1,
    /// <summary>No optimizer — frozen / target layers only. No moment vectors allocated; weight updates are no-ops.</summary>
    None = -1,
}

public enum RLLayerKind
{
    Dense    = 0,
    Dropout  = 1,
    LayerNorm = 2,
    Flatten  = 3,
}
