using System;
using Godot.Collections;
using Godot;

namespace RlAgentPlugin.Runtime;

public static class InferencePolicyFactory
{
    private static readonly System.Collections.Generic.Dictionary<string, Func<RLCheckpoint, RLNetworkGraph?, IInferencePolicy>> _customFactories =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Register a custom inference policy factory keyed by algorithm name.
    /// The <paramref name="algorithmName"/> should match the string stored in <c>RLCheckpoint.Algorithm</c>
    /// (e.g. the value your custom <see cref="ITrainer"/> writes to checkpoints via
    /// <see cref="CheckpointMetadataBuilder"/> or directly).
    /// Custom factories take priority over built-in PPO/SAC handling.
    /// </summary>
    public static void Register(string algorithmName, Func<RLCheckpoint, RLNetworkGraph?, IInferencePolicy> factory)
    {
        if (string.IsNullOrWhiteSpace(algorithmName))
            throw new ArgumentException("Algorithm name cannot be blank.", nameof(algorithmName));
        _customFactories[algorithmName.Trim()] = factory ?? throw new ArgumentNullException(nameof(factory));
    }

    /// <summary>Remove a previously registered custom inference policy factory.</summary>
    public static void Unregister(string algorithmName)
    {
        if (!string.IsNullOrWhiteSpace(algorithmName))
            _customFactories.Remove(algorithmName.Trim());
    }

    public static IInferencePolicy Create(RLCheckpoint checkpoint, RLNetworkGraph? fallbackGraph = null)
    {
        var graph = ReconstructGraph(checkpoint, fallbackGraph);

        // Custom factories take priority over built-in handlers.
        if (_customFactories.TryGetValue(checkpoint.Algorithm, out var customFactory))
            return customFactory(checkpoint, graph);

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
            var layers = new Array<Resource>();
            for (var i = 0; i < checkpoint.GraphLayerSizes.Length; i++)
            {
                layers.Add(new RLDenseLayerDef
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
