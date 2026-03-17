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

    public PolicyValueNetwork(int observationSize, int actionCount, RLNetworkGraph graph)
    {
        var useAdam = graph.Optimizer == RLOptimizerKind.Adam;
        _trunkLayers = graph.BuildTrunkLayers(observationSize);
        var lastSize = graph.OutputSize(observationSize);
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

    public BatchNetworkInference InferBatch(VectorBatch observations)
    {
        var trunkOutput = observations;
        foreach (var layer in _trunkLayers)
        {
            trunkOutput = layer.ForwardBatch(trunkOutput);
        }

        var logits = _policyHead.ForwardBatch(trunkOutput);
        var valueBatch = _valueHead.ForwardBatch(trunkOutput);
        var values = new float[observations.BatchSize];
        for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
        {
            values[batchIndex] = valueBatch.Get(batchIndex, 0);
        }

        return new BatchNetworkInference
        {
            Logits = logits,
            Values = values,
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

    public PpoBatchUpdateStats ApplyGradients(IReadOnlyList<TrainingSample> samples, RLTrainerConfig config)
    {
        if (samples.Count == 0)
        {
            return new PpoBatchUpdateStats();
        }

        var trunkGradients = new GradientBuffer[_trunkLayers.Length];
        for (var layerIndex = 0; layerIndex < _trunkLayers.Length; layerIndex++)
        {
            trunkGradients[layerIndex] = _trunkLayers[layerIndex].CreateGradientBuffer();
        }

        var policyGradients = _policyHead.CreateGradientBuffer();
        var valueGradients = _valueHead.CreateGradientBuffer();

        var totalPolicyLoss = 0f;
        var totalValueLoss = 0f;
        var totalEntropy = 0f;
        var clipCount = 0;

        foreach (var sample in samples)
        {
            var inference = Infer(sample.Observation);
            var probs = Softmax(inference.Logits);
            var actionProbability = Math.Clamp(probs[sample.Action], 1e-6f, 1.0f);
            var logProbability = Mathf.Log(actionProbability);
            var ratio = Mathf.Exp(logProbability - sample.OldLogProbability);
            var clippedRatio = Math.Clamp(ratio, 1.0f - config.ClipEpsilon, 1.0f + config.ClipEpsilon);
            var unclippedObjective = ratio * sample.Advantage;
            var clippedObjective = clippedRatio * sample.Advantage;
            var policyObjective = Math.Min(unclippedObjective, clippedObjective);
            totalPolicyLoss += -policyObjective;

            if (Mathf.Abs(ratio - 1.0f) > config.ClipEpsilon)
            {
                clipCount += 1;
            }

            var logitsGradient = new float[probs.Length];
            if (unclippedObjective <= clippedObjective)
            {
                for (var index = 0; index < probs.Length; index++)
                {
                    logitsGradient[index] = ratio * probs[index] * sample.Advantage;
                }

                logitsGradient[sample.Action] -= ratio * sample.Advantage;
            }

            var entropy = 0f;
            foreach (var probability in probs)
            {
                if (probability > 1e-6f)
                {
                    entropy -= probability * Mathf.Log(probability);
                }
            }

            totalEntropy += entropy;
            if (config.EntropyCoefficient > 0f)
            {
                for (var j = 0; j < logitsGradient.Length; j++)
                {
                    var logPj = probs[j] > 1e-6f ? Mathf.Log(probs[j]) : Mathf.Log(1e-6f);
                    logitsGradient[j] += config.EntropyCoefficient * probs[j] * (entropy + logPj);
                }
            }

            var valuePrediction = inference.Value;
            var valueError = valuePrediction - sample.Return;
            var valueLoss = valueError * valueError;
            var valueGradientScalar = valueError;

            if (config.UseValueClipping && config.ValueClipEpsilon > 0f)
            {
                var clippedValue = sample.ValueEstimate
                    + Math.Clamp(valuePrediction - sample.ValueEstimate, -config.ValueClipEpsilon, config.ValueClipEpsilon);
                var clippedError = clippedValue - sample.Return;
                var clippedValueLoss = clippedError * clippedError;
                if (clippedValueLoss > valueLoss)
                {
                    valueLoss = clippedValueLoss;
                    if (Mathf.Abs(valuePrediction - sample.ValueEstimate) > config.ValueClipEpsilon)
                    {
                        valueGradientScalar = 0f;
                    }
                }
            }

            totalValueLoss += valueLoss;
            var valueGradient = new[] { config.ValueLossCoefficient * valueGradientScalar };

            var trunkGradientFromPolicy = _policyHead.AccumulateGradients(inference.Cache.LastOutput, logitsGradient, policyGradients);
            var trunkGradientFromValue = _valueHead.AccumulateGradients(inference.Cache.LastOutput, valueGradient, valueGradients);

            var trunkGradient = new float[trunkGradientFromPolicy.Length];
            for (var index = 0; index < trunkGradient.Length; index++)
            {
                trunkGradient[index] = trunkGradientFromPolicy[index] + trunkGradientFromValue[index];
            }

            for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                var layerCache = inference.Cache.HiddenLayers[layerIndex];
                trunkGradient = _trunkLayers[layerIndex].AccumulateGradients(
                    layerCache.Input,
                    trunkGradient,
                    trunkGradients[layerIndex],
                    layerCache.PreActivation);
            }
        }

        var globalNormSquared = policyGradients.SumSquares() + valueGradients.SumSquares();
        foreach (var gradients in trunkGradients)
        {
            globalNormSquared += gradients.SumSquares();
        }

        var gradientScale = 1f / samples.Count;
        if (config.MaxGradientNorm > 0f)
        {
            var averageNorm = Mathf.Sqrt(globalNormSquared) * gradientScale;
            if (averageNorm > config.MaxGradientNorm)
            {
                gradientScale *= config.MaxGradientNorm / averageNorm;
            }
        }

        _policyHead.ApplyGradients(policyGradients, config.LearningRate, gradientScale);
        _valueHead.ApplyGradients(valueGradients, config.LearningRate, gradientScale);
        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            _trunkLayers[layerIndex].ApplyGradients(trunkGradients[layerIndex], config.LearningRate, gradientScale);
        }

        return new PpoBatchUpdateStats
        {
            PolicyLoss = totalPolicyLoss / samples.Count,
            ValueLoss = totalValueLoss / samples.Count,
            Entropy = totalEntropy / samples.Count,
            ClipFraction = (float)clipCount / samples.Count,
        };
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

    public int[] SelectGreedyActions(VectorBatch observations)
    {
        var inference = InferBatch(observations);
        var actions = new int[observations.BatchSize];
        for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
        {
            var bestIndex = 0;
            var bestValue = inference.Logits.Get(batchIndex, 0);
            for (var actionIndex = 1; actionIndex < inference.Logits.VectorSize; actionIndex++)
            {
                var logit = inference.Logits.Get(batchIndex, actionIndex);
                if (logit > bestValue)
                {
                    bestValue = logit;
                    bestIndex = actionIndex;
                }
            }

            actions[batchIndex] = bestIndex;
        }

        return actions;
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

    internal sealed class BatchNetworkInference
    {
        public VectorBatch Logits { get; init; } = new(1, 1);
        public float[] Values { get; init; } = Array.Empty<float>();
    }

    internal sealed class PpoBatchUpdateStats
    {
        public float PolicyLoss { get; init; }
        public float ValueLoss { get; init; }
        public float Entropy { get; init; }
        public float ClipFraction { get; init; }
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
