using System;
using System.Collections.Generic;
using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLNetworkGraph : Resource
{
    [Export] public Array<RLNetworkLayerDef> TrunkLayers { get; set; } = new();
    [Export] public RLOptimizerKind Optimizer { get; set; } = RLOptimizerKind.Adam;

    /// <summary>Output size of the trunk; equals the last layer's Size, or inputSize if the trunk is empty.</summary>
    public int OutputSize(int inputSize)
    {
        for (var i = TrunkLayers.Count - 1; i >= 0; i--)
        {
            if (TrunkLayers[i] is not null)
            {
                return TrunkLayers[i].Size;
            }
        }

        return inputSize;
    }

    /// <summary>Constructs trunk DenseLayer instances wired in order from inputSize.</summary>
    internal DenseLayer[] BuildTrunkLayers(int inputSize, bool? forceUseAdam = null)
    {
        if (TrunkLayers.Count == 0) return System.Array.Empty<DenseLayer>();

        var useAdam = forceUseAdam ?? Optimizer == RLOptimizerKind.Adam;
        var layers = new List<DenseLayer>(TrunkLayers.Count);
        var prev = inputSize;
        for (var i = 0; i < TrunkLayers.Count; i++)
        {
            var layerDef = TrunkLayers[i];
            if (layerDef is null)
            {
                continue;
            }

            layers.Add(new DenseLayer(prev, layerDef.Size, layerDef.Activation, useAdam));
            prev = layerDef.Size;
        }

        return layers.ToArray();
    }

    /// <summary>Returns the layer sizes as a plain int array (used for checkpoint metadata).</summary>
    public int[] GetLayerSizes()
    {
        var sizes = new int[TrunkLayers.Count];
        for (var i = 0; i < TrunkLayers.Count; i++) sizes[i] = TrunkLayers[i]?.Size ?? 0;
        return sizes;
    }

    /// <summary>Returns the layer activations as a plain int array (used for checkpoint metadata).</summary>
    public int[] GetLayerActivations()
    {
        var activations = new int[TrunkLayers.Count];
        for (var i = 0; i < TrunkLayers.Count; i++) activations[i] = (int)(TrunkLayers[i]?.Activation ?? RLActivationKind.Tanh);
        return activations;
    }

    /// <summary>Returns a sensible default network: two 64-unit Tanh layers with Adam optimizer.</summary>
    public static RLNetworkGraph CreateDefault()
    {
        return new RLNetworkGraph
        {
            TrunkLayers = new Array<RLNetworkLayerDef>
            {
                new RLNetworkLayerDef { Size = 64, Activation = RLActivationKind.Tanh },
                new RLNetworkLayerDef { Size = 64, Activation = RLActivationKind.Tanh },
            },
            Optimizer = RLOptimizerKind.Adam,
        };
    }
}
