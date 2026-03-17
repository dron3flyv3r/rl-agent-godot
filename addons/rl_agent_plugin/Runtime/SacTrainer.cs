using System;

namespace RlAgentPlugin.Runtime;

public sealed class SacTrainer : ITrainer
{
    private readonly PolicyGroupConfig _config;
    private readonly RLTrainerConfig _trainerConfig;
    private readonly SacNetwork _network;
    private readonly SacReplayBuffer _buffer;
    private readonly Random _rng;
    private readonly bool _isContinuous;

    private float _logAlpha;
    private readonly float _targetEntropy;
    private long _totalStepsSeen;

    public SacTrainer(PolicyGroupConfig config)
    {
        _config = config;
        _trainerConfig = config.TrainerConfig;
        _isContinuous = config.ContinuousActionDimensions > 0 && config.DiscreteActionCount == 0;
        _rng = new Random();

        _network = new SacNetwork(
            config.ObservationSize,
            _isContinuous ? config.ContinuousActionDimensions : config.DiscreteActionCount,
            _isContinuous,
            config.NetworkGraph,
            config.TrainerConfig.LearningRate);

        _buffer = new SacReplayBuffer(config.TrainerConfig.ReplayBufferCapacity);

        _logAlpha = MathF.Log(Math.Max(config.TrainerConfig.SacInitAlpha, 1e-8f));

        // Target entropy: discrete → log(|A|); continuous → -action_dims
        _targetEntropy = _isContinuous
            ? -config.ContinuousActionDimensions
            : MathF.Log(config.DiscreteActionCount);
    }

    public PolicyDecision SampleAction(float[] observation)
    {
        if (_isContinuous)
        {
            var (action, logProb, _, _) = _network.SampleContinuousAction(observation, _rng);
            return new PolicyDecision
            {
                DiscreteAction = -1,
                ContinuousActions = action,
                LogProbability = logProb,
                Value = 0f,
                Entropy = -logProb,
            };
        }
        else
        {
            var (action, logProb, entropy) = _network.SampleDiscreteAction(observation, _rng);
            return new PolicyDecision
            {
                DiscreteAction = action,
                ContinuousActions = Array.Empty<float>(),
                LogProbability = logProb,
                Value = 0f,
                Entropy = entropy,
            };
        }
    }

    public PolicyDecision[] SampleActions(VectorBatch observations)
    {
        var decisions = new PolicyDecision[observations.BatchSize];
        if (_isContinuous)
        {
            var batch = _network.SampleContinuousActions(observations, _rng);
            for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
            {
                decisions[batchIndex] = new PolicyDecision
                {
                    DiscreteAction = -1,
                    ContinuousActions = batch.Actions[batchIndex],
                    LogProbability = batch.LogProbabilities[batchIndex],
                    Value = 0f,
                    Entropy = -batch.LogProbabilities[batchIndex],
                };
            }

            return decisions;
        }

        var discreteBatch = _network.SampleDiscreteActions(observations, _rng);
        for (var batchIndex = 0; batchIndex < observations.BatchSize; batchIndex++)
        {
            decisions[batchIndex] = new PolicyDecision
            {
                DiscreteAction = discreteBatch.Actions[batchIndex],
                ContinuousActions = Array.Empty<float>(),
                LogProbability = discreteBatch.LogProbabilities[batchIndex],
                Value = 0f,
                Entropy = discreteBatch.Entropies[batchIndex],
            };
        }

        return decisions;
    }

    public float EstimateValue(float[] observation) => 0f;

    public float[] EstimateValues(VectorBatch observations) => new float[observations.BatchSize];

    public void RecordTransition(Transition transition)
    {
        _buffer.Add(transition);
        _totalStepsSeen++;
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_buffer.Count < _trainerConfig.SacWarmupSteps)
        {
            return null;
        }

        if (_totalStepsSeen % Math.Max(1, _trainerConfig.SacUpdateEverySteps) != 0)
        {
            return null;
        }

        var batch = _buffer.SampleBatch(_trainerConfig.SacBatchSize, _rng);
        var alpha = MathF.Exp(_logAlpha);

        var policyLoss = 0f;
        var valueLoss = 0f;
        var entropySum = 0f;

        foreach (var t in batch)
        {
            if (_isContinuous)
            {
                UpdateContinuous(t, alpha, ref policyLoss, ref valueLoss, ref entropySum);
            }
            else
            {
                UpdateDiscrete(t, alpha, ref policyLoss, ref valueLoss, ref entropySum);
            }
        }

        _network.SoftUpdateTargets(_trainerConfig.SacTau);

        var n = Math.Max(1, batch.Length);
        return new TrainerUpdateStats
        {
            PolicyLoss = policyLoss / n,
            ValueLoss = valueLoss / n,
            Entropy = entropySum / n,
            ClipFraction = 0f,
            Checkpoint = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public RLCheckpoint CreateCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        return CheckpointMetadataBuilder.Apply(
            _network.SaveCheckpoint(groupId, totalSteps, episodeCount, updateCount),
            _config);
    }

    // ── Discrete SAC update ──────────────────────────────────────────────────

    private void UpdateDiscrete(Transition t, float alpha, ref float policyLoss, ref float valueLoss, ref float entropySum)
    {
        // Compute target V(s') = sum_a pi(a|s') * (min_Q_target(s',a) - alpha * log_pi(a|s'))
        var (nextQ1, nextQ2) = _network.GetDiscreteTargetQValues(t.NextObservation);
        var (nextProbs, nextLogProbs) = DiscreteActorProbs(t.NextObservation);

        var nextV = 0f;
        for (var a = 0; a < nextQ1.Length; a++)
        {
            var minQ = Math.Min(nextQ1[a], nextQ2[a]);
            nextV += nextProbs[a] * (minQ - alpha * nextLogProbs[a]);
        }

        // Bellman target
        var mask = t.Done ? 0f : 1f;
        var y = t.Reward + _trainerConfig.Gamma * mask * nextV;

        // Update Q networks
        var (q1, q2) = _network.GetDiscreteQValues(t.Observation);
        valueLoss += MathF.Abs(q1[t.DiscreteAction] - y) + MathF.Abs(q2[t.DiscreteAction] - y);

        _network.UpdateQ1Discrete(t.Observation, t.DiscreteAction, y);
        _network.UpdateQ2Discrete(t.Observation, t.DiscreteAction, y);

        // Update actor
        _network.UpdateActorDiscrete(t.Observation, alpha);

        // Compute entropy for logging and alpha update
        var (probs, logProbs) = DiscreteActorProbs(t.Observation);
        var entropy = 0f;
        for (var a = 0; a < probs.Length; a++)
        {
            entropy -= probs[a] * logProbs[a];
        }

        entropySum += entropy;

        // Auto-tune alpha: minimize log_alpha * (entropy - target_entropy)
        if (_trainerConfig.SacAutoTuneAlpha)
        {
            var alphaLossGrad = -(entropy - _targetEntropy);
            _logAlpha -= _trainerConfig.LearningRate * alphaLossGrad;
        }

        policyLoss -= entropy; // policy loss ≈ -entropy as a proxy metric
    }

    // ── Continuous SAC update ────────────────────────────────────────────────

    private void UpdateContinuous(Transition t, float alpha, ref float policyLoss, ref float valueLoss, ref float entropySum)
    {
        // Sample next action for target
        var (nextAction, nextLogProb, _, _) = _network.SampleContinuousAction(t.NextObservation, _rng);
        var (nextQ1t, nextQ2t) = _network.GetContinuousTargetQValues(t.NextObservation, nextAction);
        var nextMinQ = Math.Min(nextQ1t, nextQ2t);
        var nextV = nextMinQ - alpha * nextLogProb;

        var mask = t.Done ? 0f : 1f;
        var y = t.Reward + _trainerConfig.Gamma * mask * nextV;

        // Update Q networks
        var (q1, q2) = _network.GetContinuousQValues(t.Observation, t.ContinuousActions);
        valueLoss += MathF.Abs(q1 - y) + MathF.Abs(q2 - y);

        _network.UpdateQ1Continuous(t.Observation, t.ContinuousActions, y);
        _network.UpdateQ2Continuous(t.Observation, t.ContinuousActions, y);

        // Sample fresh action for actor update (reparameterization)
        var (action, logProb, eps, u) = _network.SampleContinuousAction(t.Observation, _rng);
        _network.UpdateActorContinuous(t.Observation, action, eps, u, alpha);

        entropySum += -logProb;

        // Auto-tune alpha
        if (_trainerConfig.SacAutoTuneAlpha)
        {
            var alphaLossGrad = -(-logProb - _targetEntropy);
            _logAlpha -= _trainerConfig.LearningRate * alphaLossGrad;
        }

        policyLoss += logProb; // policy loss proxy
    }

    private (float[] probs, float[] logProbs) DiscreteActorProbs(float[] obs)
    {
        return _network.GetDiscreteActorProbs(obs);
    }
}
