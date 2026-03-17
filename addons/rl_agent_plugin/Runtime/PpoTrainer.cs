using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class PpoTrainer : ITrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly PolicyValueNetwork _network;
    private readonly List<PpoTransition> _transitions = new();
    private readonly RandomNumberGenerator _rng = new();

    public PpoTrainer(PolicyGroupConfig config)
    {
        _config = config;
        _trainerConfig = config.TrainerConfig;
        _network = new PolicyValueNetwork(config.ObservationSize, config.DiscreteActionCount, config.NetworkConfig);
        _rng.Randomize();
    }

    public int TransitionCount => _transitions.Count;

    public PolicyDecision SampleAction(float[] observation)
    {
        var inference = _network.Infer(observation);
        var probabilities = Softmax(inference.Logits);
        var sampledAction = SampleFromProbabilities(probabilities);
        var logProbability = Mathf.Log(Math.Max(1e-6f, probabilities[sampledAction]));

        return new PolicyDecision
        {
            DiscreteAction = sampledAction,
            Value = inference.Value,
            LogProbability = logProbability,
            Entropy = CalculateEntropy(probabilities),
        };
    }

    public float EstimateValue(float[] observation)
    {
        return _network.Infer(observation).Value;
    }

    public void RecordTransition(Transition t)
    {
        _transitions.Add(new PpoTransition
        {
            Observation = t.Observation.ToArray(),
            Action = t.DiscreteAction,
            Reward = t.Reward,
            Done = t.Done,
            OldLogProbability = t.OldLogProbability,
            Value = t.Value,
            NextValue = t.NextValue,
        });
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_transitions.Count < _trainerConfig.RolloutLength)
        {
            return null;
        }

        var samples = BuildTrainingSamples();
        var policyLoss = 0.0f;
        var valueLoss = 0.0f;
        var entropy = 0.0f;
        foreach (var sample in samples)
        {
            policyLoss += -sample.Advantage;
            valueLoss += Mathf.Abs(sample.Return - sample.ValueEstimate);
        }

        NormalizeAdvantages(samples);

        for (var epoch = 0; epoch < _trainerConfig.EpochsPerUpdate; epoch++)
        {
            foreach (var sample in samples)
            {
                _network.ApplyGradients(sample, _trainerConfig);
            }
        }

        _transitions.Clear();

        return new TrainerUpdateStats
        {
            PolicyLoss = policyLoss / samples.Count,
            ValueLoss = valueLoss / samples.Count,
            Entropy = entropy / samples.Count,
            Checkpoint = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        return CheckpointMetadataBuilder.Apply(
            _network.SaveCheckpoint(groupId, totalSteps, episodeCount, updateCount),
            _config);
    }

    private List<TrainingSample> BuildTrainingSamples()
    {
        var samples = new List<TrainingSample>(_transitions.Count);
        var advantages = new float[_transitions.Count];
        var returns = new float[_transitions.Count];
        var nextAdvantage = 0.0f;

        for (var index = _transitions.Count - 1; index >= 0; index--)
        {
            var transition = _transitions[index];
            var mask = transition.Done ? 0.0f : 1.0f;
            var delta = transition.Reward + (_trainerConfig.Gamma * transition.NextValue * mask) - transition.Value;
            nextAdvantage = delta + (_trainerConfig.Gamma * _trainerConfig.GaeLambda * mask * nextAdvantage);
            advantages[index] = nextAdvantage;
            returns[index] = transition.Value + advantages[index];
        }

        for (var index = 0; index < _transitions.Count; index++)
        {
            var transition = _transitions[index];
            samples.Add(new TrainingSample
            {
                Observation = transition.Observation,
                Action = transition.Action,
                Return = returns[index],
                Advantage = advantages[index],
                OldLogProbability = transition.OldLogProbability,
                ValueEstimate = transition.Value,
            });
        }

        return samples;
    }

    private static void NormalizeAdvantages(IList<TrainingSample> samples)
    {
        var mean = samples.Average(sample => sample.Advantage);
        var variance = samples.Average(sample =>
        {
            var diff = sample.Advantage - mean;
            return diff * diff;
        });

        var stdDev = Mathf.Sqrt((float)variance + 1e-8f);
        foreach (var sample in samples)
        {
            sample.Advantage = (sample.Advantage - (float)mean) / stdDev;
        }
    }

    private int SampleFromProbabilities(float[] probabilities)
    {
        var roll = _rng.Randf();
        var cumulative = 0.0f;
        for (var index = 0; index < probabilities.Length; index++)
        {
            cumulative += probabilities[index];
            if (roll <= cumulative)
            {
                return index;
            }
        }

        return probabilities.Length - 1;
    }

    private static float[] Softmax(IReadOnlyList<float> logits)
    {
        var maxLogit = logits.Max();
        var probabilities = new float[logits.Count];
        var total = 0.0f;
        for (var index = 0; index < logits.Count; index++)
        {
            probabilities[index] = Mathf.Exp(logits[index] - maxLogit);
            total += probabilities[index];
        }

        for (var index = 0; index < probabilities.Length; index++)
        {
            probabilities[index] /= total;
        }

        return probabilities;
    }

    private static float CalculateEntropy(IReadOnlyList<float> probabilities)
    {
        var entropy = 0.0f;
        foreach (var probability in probabilities)
        {
            if (probability <= 1e-6f)
            {
                continue;
            }

            entropy -= probability * Mathf.Log(probability);
        }

        return entropy;
    }

    private sealed class PpoTransition
    {
        public float[] Observation { get; init; } = Array.Empty<float>();
        public int Action { get; init; }
        public float Reward { get; init; }
        public bool Done { get; init; }
        public float OldLogProbability { get; init; }
        public float Value { get; init; }
        public float NextValue { get; init; }
    }
}

public sealed class TrainingSample
{
    public float[] Observation { get; init; } = Array.Empty<float>();
    public int Action { get; init; }
    public float Return { get; init; }
    public float Advantage { get; set; }
    public float OldLogProbability { get; init; }
    public float ValueEstimate { get; init; }
}
