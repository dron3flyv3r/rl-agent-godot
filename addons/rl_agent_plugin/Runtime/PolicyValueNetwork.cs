using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class PolicyValueNetwork
{
    private readonly DenseLayer[] _trunkLayers;
    private readonly DenseLayer _policyHead;
    private readonly DenseLayer _valueHead;

    public PolicyValueNetwork(int observationSize, int actionCount, RLNetworkConfig config)
    {
        var useAdam = config.Optimizer == RLOptimizerKind.Adam;

        var layerSizes = new List<int> { observationSize };
        layerSizes.AddRange(config.HiddenLayerSizes.Where(size => size > 0));

        _trunkLayers = new DenseLayer[Math.Max(0, layerSizes.Count - 1)];
        for (var index = 0; index < layerSizes.Count - 1; index++)
        {
            _trunkLayers[index] = new DenseLayer(layerSizes[index], layerSizes[index + 1], config.Activation, useAdam);
        }

        var lastSize = layerSizes[^1];
        _policyHead = new DenseLayer(lastSize, actionCount, null, useAdam);
        _valueHead = new DenseLayer(lastSize, 1, null, useAdam);
    }

    public NetworkInference Infer(float[] observation)
    {
        var cache = new ForwardCache(observation);
        foreach (var layer in _trunkLayers)
        {
            cache.AddLayer(layer.Forward(cache.LastOutput));
        }

        cache.Policy = _policyHead.Forward(cache.LastOutput);
        cache.Value = _valueHead.Forward(cache.LastOutput);

        return new NetworkInference
        {
            Cache = cache,
            Logits = cache.Policy.Activated,
            Value = cache.Value.Activated[0],
        };
    }

    public void ApplyGradients(TrainingSample sample, RLTrainerConfig config)
    {
        var inference = Infer(sample.Observation);
        var probs = Softmax(inference.Logits);
        var actionProbability = Math.Clamp(probs[sample.Action], 1e-6f, 1.0f);
        var ratio = Mathf.Exp(Mathf.Log(actionProbability) - sample.OldLogProbability);
        var unclippedObjective = ratio * sample.Advantage;
        var clippedRatio = Math.Clamp(ratio, 1.0f - config.ClipEpsilon, 1.0f + config.ClipEpsilon);
        var clippedObjective = clippedRatio * sample.Advantage;

        var logitsGradient = new float[probs.Length];
        if (unclippedObjective <= clippedObjective)
        {
            for (var index = 0; index < probs.Length; index++)
            {
                logitsGradient[index] = ratio * probs[index] * sample.Advantage;
            }

            logitsGradient[sample.Action] -= ratio * sample.Advantage;
        }

        // Entropy bonus: add entropyCoeff * H to the objective.
        // Gradient of -entropyCoeff*H w.r.t. logit j:
        //   = -entropyCoeff * dH/dz_j
        //   = -entropyCoeff * p_j * (-H - log(p_j))
        //   = entropyCoeff * p_j * (H + log(p_j))
        if (config.EntropyCoefficient > 0f)
        {
            var entropy = 0f;
            foreach (var p in probs)
            {
                if (p > 1e-6f) entropy -= p * Mathf.Log(p);
            }

            for (var j = 0; j < logitsGradient.Length; j++)
            {
                var logPj = probs[j] > 1e-6f ? Mathf.Log(probs[j]) : Mathf.Log(1e-6f);
                logitsGradient[j] += config.EntropyCoefficient * probs[j] * (entropy + logPj);
            }
        }

        var valueError = inference.Value - sample.Return;
        var valueGradient = new[] { config.ValueLossCoefficient * valueError };

        var trunkGradientFromPolicy = _policyHead.Backward(inference.Cache.LastOutput, logitsGradient, config.LearningRate);
        var trunkGradientFromValue = _valueHead.Backward(inference.Cache.LastOutput, valueGradient, config.LearningRate);

        var trunkGradient = new float[trunkGradientFromPolicy.Length];
        for (var index = 0; index < trunkGradient.Length; index++)
        {
            trunkGradient[index] = trunkGradientFromPolicy[index] + trunkGradientFromValue[index];
        }

        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            var layerCache = inference.Cache.HiddenLayers[layerIndex];
            trunkGradient = _trunkLayers[layerIndex].Backward(layerCache.Input, trunkGradient, config.LearningRate, layerCache.PreActivation);
        }
    }

    public RLCheckpoint SaveCheckpoint(string runId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes = new List<int>();
        foreach (var layer in _trunkLayers)
        {
            layer.AppendSerialized(weights, shapes);
        }

        _policyHead.AppendSerialized(weights, shapes);
        _valueHead.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId = runId,
            TotalSteps = totalSteps,
            EpisodeCount = episodeCount,
            UpdateCount = updateCount,
            WeightBuffer = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var weightIndex = 0;
        var shapeIndex = 0;

        foreach (var layer in _trunkLayers)
        {
            layer.LoadSerialized(checkpoint.WeightBuffer, ref weightIndex, checkpoint.LayerShapeBuffer, ref shapeIndex);
        }

        _policyHead.LoadSerialized(checkpoint.WeightBuffer, ref weightIndex, checkpoint.LayerShapeBuffer, ref shapeIndex);
        _valueHead.LoadSerialized(checkpoint.WeightBuffer, ref weightIndex, checkpoint.LayerShapeBuffer, ref shapeIndex);
    }

    public int SelectGreedyAction(float[] observation)
    {
        var logits = Infer(observation).Logits;
        var bestIndex = 0;
        var bestValue = logits[0];
        for (var index = 1; index < logits.Length; index++)
        {
            if (logits[index] > bestValue)
            {
                bestValue = logits[index];
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expValues = new float[logits.Length];
        var total = 0.0f;
        for (var index = 0; index < logits.Length; index++)
        {
            expValues[index] = Mathf.Exp(logits[index] - maxLogit);
            total += expValues[index];
        }

        for (var index = 0; index < expValues.Length; index++)
        {
            expValues[index] /= total;
        }

        return expValues;
    }

    internal sealed class NetworkInference
    {
        public ForwardCache Cache { get; init; } = new(Array.Empty<float>());
        public float[] Logits { get; init; } = Array.Empty<float>();
        public float Value { get; init; }
    }

    internal sealed class ForwardCache
    {
        public ForwardCache(float[] observation)
        {
            Observation = observation;
        }

        public float[] Observation { get; }
        public List<LayerCache> HiddenLayers { get; } = new();
        public LayerCache Policy { get; set; } = LayerCache.Empty;
        public LayerCache Value { get; set; } = LayerCache.Empty;
        public float[] LastOutput => HiddenLayers.Count == 0 ? Observation : HiddenLayers[^1].Activated;

        public void AddLayer(LayerCache cache)
        {
            HiddenLayers.Add(cache);
        }
    }
}
