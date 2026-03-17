using System;

namespace RlAgentPlugin.Runtime;

public static class InferencePolicyFactory
{
    public static IInferencePolicy Create(RLCheckpoint checkpoint, RLNetworkConfig? fallbackConfig = null)
    {
        var networkConfig = CreateNetworkConfig(checkpoint, fallbackConfig);
        return string.Equals(checkpoint.Algorithm, RLCheckpoint.SacAlgorithm, StringComparison.OrdinalIgnoreCase)
            ? new SacInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.ContinuousActionDimensions > 0
                    ? checkpoint.ContinuousActionDimensions
                    : checkpoint.DiscreteActionCount,
                checkpoint.ContinuousActionDimensions > 0,
                networkConfig)
            : new PpoInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.DiscreteActionCount,
                networkConfig);
    }

    private static RLNetworkConfig CreateNetworkConfig(RLCheckpoint checkpoint, RLNetworkConfig? fallbackConfig)
    {
        var hiddenLayerSizes = checkpoint.HiddenLayerSizes.Length > 0
            ? (int[])checkpoint.HiddenLayerSizes.Clone()
            : fallbackConfig?.HiddenLayerSizes ?? Array.Empty<int>();

        return new RLNetworkConfig
        {
            HiddenLayerSizes = hiddenLayerSizes,
            Activation = checkpoint.InferHiddenActivation(fallbackConfig?.Activation ?? RLActivationKind.Tanh),
            Optimizer = fallbackConfig?.Optimizer ?? RLOptimizerKind.Adam,
            SharedTrunk = fallbackConfig?.SharedTrunk ?? true,
        };
    }
}
