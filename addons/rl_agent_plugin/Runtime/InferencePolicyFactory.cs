using System;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

public static class InferencePolicyFactory
{
    public static IInferencePolicy Create(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph = null)
    {
        var graph = ReconstructGraph(checkpoint, fallbackGraph);
        return string.Equals(checkpoint.Algorithm, RLCheckpoint.SacAlgorithm, StringComparison.OrdinalIgnoreCase)
            ? new SacInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.ContinuousActionDimensions > 0
                    ? checkpoint.ContinuousActionDimensions
                    : checkpoint.DiscreteActionCount,
                checkpoint.ContinuousActionDimensions > 0,
                graph)
            : new PpoInferencePolicy(
                checkpoint.ObservationSize,
                checkpoint.DiscreteActionCount,
                graph);
    }

    /// <summary>
    /// Rebuilds an <see cref="RLNetworkGraph"/> from checkpoint metadata.
    /// Prefers explicitly stored graph fields; falls back to the provided graph when none are present.
    /// </summary>
    private static RLNetworkGraph ReconstructGraph(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph)
    {
        if (checkpoint.GraphLayerSizes.Length > 0)
        {
            var layers = new Array<RLNetworkLayerDef>();
            for (var i = 0; i < checkpoint.GraphLayerSizes.Length; i++)
            {
                layers.Add(new RLNetworkLayerDef
                {
                    Size = checkpoint.GraphLayerSizes[i],
                    Activation = checkpoint.GraphLayerActivations.Length > i
                        ? (RLActivationKind)checkpoint.GraphLayerActivations[i]
                        : RLActivationKind.Tanh,
                });
            }

            return new RLNetworkGraph
            {
                TrunkLayers = layers,
                Optimizer = (RLOptimizerKind)checkpoint.GraphOptimizer,
            };
        }

        if (fallbackGraph is not null) return fallbackGraph;

        return new RLNetworkGraph();
    }
}
