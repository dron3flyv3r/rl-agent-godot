using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

internal sealed class PolicyValueNetwork
{
    private readonly NetworkLayer[] _trunkLayers;
    private readonly DenseLayer _policyHead;
    private readonly DenseLayer _valueHead;

    public PolicyValueNetwork(int observationSize, int actionCount, RLNetworkGraph graph)
    {
        _trunkLayers = graph.BuildTrunkLayers(observationSize);
        var lastSize = graph.OutputSize(observationSize);
        _policyHead = new DenseLayer(lastSize, actionCount, null, graph.Optimizer);
        _valueHead  = new DenseLayer(lastSize, 1,           null, graph.Optimizer);
    }

    public NetworkInference Infer(float[] observation)
    {
        var x = observation;
        foreach (var layer in _trunkLayers)
            x = layer.Forward(x);

        var logits = _policyHead.Forward(x);
        var value  = _valueHead.Forward(x);

        return new NetworkInference { Logits = logits, Value = value[0] };
    }

    public BatchNetworkInference InferBatch(VectorBatch observations)
    {
        var trunkOutput = observations;
        foreach (var layer in _trunkLayers)
            trunkOutput = layer.ForwardBatch(trunkOutput);

        var logits     = _policyHead.ForwardBatch(trunkOutput);
        var valueBatch = _valueHead.ForwardBatch(trunkOutput);
        var values     = new float[observations.BatchSize];
        for (var b = 0; b < observations.BatchSize; b++)
            values[b] = valueBatch.Get(b, 0);

        return new BatchNetworkInference { Logits = logits, Values = values };
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
                logitsGradient[index] = ratio * probs[index] * sample.Advantage;
            logitsGradient[sample.Action] -= ratio * sample.Advantage;
        }

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

        // Layers cache their own Forward state — Backward uses it automatically.
        var trunkGradientFromPolicy = _policyHead.Backward(logitsGradient, config.LearningRate);
        var trunkGradientFromValue  = _valueHead.Backward(valueGradient,   config.LearningRate);

        var trunkGradient = new float[trunkGradientFromPolicy.Length];
        for (var index = 0; index < trunkGradient.Length; index++)
            trunkGradient[index] = trunkGradientFromPolicy[index] + trunkGradientFromValue[index];

        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
            trunkGradient = _trunkLayers[layerIndex].Backward(trunkGradient, config.LearningRate);
    }

    public PpoBatchUpdateStats ApplyGradients(IReadOnlyList<TrainingSample> samples, RLTrainerConfig config)
    {
        if (samples.Count == 0)
            return new PpoBatchUpdateStats();

        var trunkGradients = new GradientBuffer[_trunkLayers.Length];
        for (var i = 0; i < _trunkLayers.Length; i++)
            trunkGradients[i] = _trunkLayers[i].CreateGradientBuffer();

        var policyGradients = _policyHead.CreateGradientBuffer();
        var valueGradients  = _valueHead.CreateGradientBuffer();

        var totalPolicyLoss = 0f;
        var totalValueLoss  = 0f;
        var totalEntropy    = 0f;
        var clipCount       = 0;

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
            totalPolicyLoss += -Math.Min(unclippedObjective, clippedObjective);

            if (Mathf.Abs(ratio - 1.0f) > config.ClipEpsilon) clipCount++;

            var logitsGradient = new float[probs.Length];
            if (unclippedObjective <= clippedObjective)
            {
                for (var index = 0; index < probs.Length; index++)
                    logitsGradient[index] = ratio * probs[index] * sample.Advantage;
                logitsGradient[sample.Action] -= ratio * sample.Advantage;
            }

            var entropy = 0f;
            foreach (var probability in probs)
            {
                if (probability > 1e-6f) entropy -= probability * Mathf.Log(probability);
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
                        valueGradientScalar = 0f;
                }
            }

            totalValueLoss += valueLoss;
            var valueGradient = new[] { config.ValueLossCoefficient * valueGradientScalar };

            // Infer() already cached state in each layer; AccumulateGradients uses it.
            var trunkGradientFromPolicy = _policyHead.AccumulateGradients(logitsGradient, policyGradients);
            var trunkGradientFromValue  = _valueHead.AccumulateGradients(valueGradient,   valueGradients);

            var trunkGradient = new float[trunkGradientFromPolicy.Length];
            for (var index = 0; index < trunkGradient.Length; index++)
                trunkGradient[index] = trunkGradientFromPolicy[index] + trunkGradientFromValue[index];

            for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
                trunkGradient = _trunkLayers[layerIndex].AccumulateGradients(trunkGradient, trunkGradients[layerIndex]);
        }

        var globalNormSquared = policyGradients.SumSquares() + valueGradients.SumSquares();
        foreach (var g in trunkGradients) globalNormSquared += g.SumSquares();

        var gradientScale = 1f / samples.Count;
        if (config.MaxGradientNorm > 0f)
        {
            var averageNorm = Mathf.Sqrt(globalNormSquared) * gradientScale;
            if (averageNorm > config.MaxGradientNorm)
                gradientScale *= config.MaxGradientNorm / averageNorm;
        }

        _policyHead.ApplyGradients(policyGradients, config.LearningRate, gradientScale);
        _valueHead.ApplyGradients(valueGradients,   config.LearningRate, gradientScale);
        for (var layerIndex = _trunkLayers.Length - 1; layerIndex >= 0; layerIndex--)
            _trunkLayers[layerIndex].ApplyGradients(trunkGradients[layerIndex], config.LearningRate, gradientScale);

        return new PpoBatchUpdateStats
        {
            PolicyLoss    = totalPolicyLoss / samples.Count,
            ValueLoss     = totalValueLoss  / samples.Count,
            Entropy       = totalEntropy    / samples.Count,
            ClipFraction  = (float)clipCount / samples.Count,
        };
    }

    public RLCheckpoint SaveCheckpoint(string runId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes  = new List<int>();
        foreach (var layer in _trunkLayers) layer.AppendSerialized(weights, shapes);
        _policyHead.AppendSerialized(weights, shapes);
        _valueHead.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId        = runId,
            TotalSteps   = totalSteps,
            EpisodeCount = episodeCount,
            UpdateCount  = updateCount,
            WeightBuffer = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi       = 0;
        var si       = 0;
        var isLegacy = checkpoint.FormatVersion < RLCheckpoint.CurrentFormatVersion;

        foreach (var layer in _trunkLayers)
            layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);

        _policyHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
        _valueHead.LoadSerialized(checkpoint.WeightBuffer,  ref wi, checkpoint.LayerShapeBuffer, ref si, isLegacy);
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
        for (var b = 0; b < observations.BatchSize; b++)
        {
            var bestIndex = 0;
            var bestValue = inference.Logits.Get(b, 0);
            for (var a = 1; a < inference.Logits.VectorSize; a++)
            {
                var logit = inference.Logits.Get(b, a);
                if (logit > bestValue) { bestValue = logit; bestIndex = a; }
            }

            actions[b] = bestIndex;
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
            expValues[index] /= total;

        return expValues;
    }

    internal sealed class NetworkInference
    {
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
        public float PolicyLoss   { get; init; }
        public float ValueLoss    { get; init; }
        public float Entropy      { get; init; }
        public float ClipFraction { get; init; }
    }
}
