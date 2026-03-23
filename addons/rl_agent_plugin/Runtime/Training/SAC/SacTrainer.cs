using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;

namespace RlAgentPlugin.Runtime;

public sealed class SacTrainer : ITrainer, IAsyncTrainer, IDistributedTrainer
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
    // Threshold-based cadence: train when _totalStepsSeen >= this value.
    // Avoids the fragile modulo check that can skip multiples when N agents
    // make decisions per frame and N doesn't divide SacUpdateEverySteps.
    private long _nextTrainingStep;

    // ── Async gradient update state ───────────────────────────────────────────
    private SacNetwork? _shadowNetwork;
    private Task<SacAsyncResult>? _pendingUpdate;

    // ── Distributed staging buffer ────────────────────────────────────────────
    // Transitions accumulated since last worker send.
    // Capped at 2 × SacBatchSize to bound memory when not consumed by a master.
    private readonly Queue<Transition> _stagingTransitions = new();

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
        // Guard against NaN/infinity from physics explosions corrupting the replay buffer.
        if (!IsFiniteTransition(transition)) return;

        _buffer.Add(transition);
        _totalStepsSeen++;
        _stagingTransitions.Enqueue(transition);
        // Cap staging buffer so it doesn't grow forever in standalone mode
        while (_stagingTransitions.Count > _trainerConfig.SacBatchSize * 2)
            _stagingTransitions.Dequeue();
    }

    private static bool IsFiniteTransition(Transition t)
    {
        if (!float.IsFinite(t.Reward)) return false;
        foreach (var f in t.Observation)    if (!float.IsFinite(f)) return false;
        foreach (var f in t.NextObservation) if (!float.IsFinite(f)) return false;
        return true;
    }

    public TrainerUpdateStats? TryUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_buffer.Count < _trainerConfig.SacWarmupSteps)
            return null;

        if (_totalStepsSeen < _nextTrainingStep)
            return null;
        _nextTrainingStep = _totalStepsSeen + Math.Max(1, _trainerConfig.SacUpdateEverySteps);

        var batch = _buffer.SampleBatch(_trainerConfig.SacBatchSize, _rng);
        var alpha = MathF.Exp(_logAlpha);
        var policyLoss = 0f;
        var valueLoss = 0f;
        var entropySum = 0f;

        foreach (var t in batch)
        {
            if (_isContinuous)
                UpdateContinuous(_network, t, alpha, _trainerConfig, ref policyLoss, ref valueLoss, ref entropySum, _rng);
            else
                UpdateDiscrete(_network, t, alpha, _trainerConfig, ref policyLoss, ref valueLoss, ref entropySum);
        }

        // Single per-batch logAlpha update (batch-average gradient, not per-transition).
        // Sign: H < H_target → logAlpha increases (push toward more entropy).
        if (_trainerConfig.SacAutoTuneAlpha && batch.Length > 0)
        {
            var meanEntropy = entropySum / batch.Length;
            _logAlpha -= _trainerConfig.LearningRate * (meanEntropy - _targetEntropy);
            _logAlpha = Math.Clamp(_logAlpha, -20f, 4f);
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

    // ── IAsyncTrainer ─────────────────────────────────────────────────────────

    public bool TryScheduleBackgroundUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is not null)
            return false;

        if (_buffer.Count < _trainerConfig.SacWarmupSteps)
            return false;

        if (_totalStepsSeen < _nextTrainingStep)
            return false;
        _nextTrainingStep = _totalStepsSeen + Math.Max(1, _trainerConfig.SacUpdateEverySteps);

        // Pre-sample the batch on the main thread (buffer access is main-thread-safe here).
        var batch = _buffer.SampleBatch(_trainerConfig.SacBatchSize, _rng);

        // Lazy-create shadow with identical architecture.
        _shadowNetwork ??= new SacNetwork(
            _config.ObservationSize,
            _isContinuous ? _config.ContinuousActionDimensions : _config.DiscreteActionCount,
            _isContinuous,
            _config.NetworkGraph,
            _trainerConfig.LearningRate);

        // Copy live weights into shadow so the background job works on an isolated copy.
        _network.CopyWeightsTo(_shadowNetwork);

        var shadow = _shadowNetwork;
        var config = _trainerConfig;
        var isContinuous = _isContinuous;
        var logAlpha = _logAlpha;
        var targetEntropy = _targetEntropy;
        var rng = new Random(); // dedicated RNG — System.Random is not thread-safe

        _pendingUpdate = Task.Run(() => RunBackgroundUpdate(shadow, batch, config, isContinuous, logAlpha, targetEntropy, rng));
        return true;
    }

    public TrainerUpdateStats? TryPollResult(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null || !_pendingUpdate.IsCompleted)
            return null;

        if (_pendingUpdate.IsFaulted)
        {
            // Clear the stuck task so training can resume on the next eligible step.
            // (Re-throwing would leave _trainingInProgress dirty in the master, blocking forever.)
            var msg = _pendingUpdate.Exception?.GetBaseException().Message ?? "unknown";
            _pendingUpdate = null;
            Godot.GD.PushError($"[SAC] Background training task faulted — skipping update. {msg}");
            return null;
        }

        var result = _pendingUpdate.Result;
        _pendingUpdate = null;

        // Apply trained shadow weights back to the live network (main thread only).
        _network.LoadWeightsFrom(_shadowNetwork!);
        _logAlpha = result.NewLogAlpha;

        return new TrainerUpdateStats
        {
            PolicyLoss = result.PolicyLoss,
            ValueLoss = result.ValueLoss,
            Entropy = result.Entropy,
            ClipFraction = 0f,
            Checkpoint = CreateCheckpoint(groupId, totalSteps, episodeCount, 0),
        };
    }

    public TrainerUpdateStats? FlushPendingUpdate(string groupId, long totalSteps, long episodeCount)
    {
        if (_pendingUpdate is null)
            return null;
        _pendingUpdate.Wait();
        return TryPollResult(groupId, totalSteps, episodeCount);
    }

    // ── Background job ────────────────────────────────────────────────────────

    private static SacAsyncResult RunBackgroundUpdate(
        SacNetwork network,
        Transition[] batch,
        RLTrainerConfig config,
        bool isContinuous,
        float logAlpha,
        float targetEntropy,
        Random rng)
    {
        // alpha is fixed for the entire batch — only updated once after the loop.
        var alpha = MathF.Exp(logAlpha);
        var policyLoss = 0f;
        var valueLoss = 0f;
        var entropySum = 0f;

        foreach (var t in batch)
        {
            if (isContinuous)
                UpdateContinuous(network, t, alpha, config, ref policyLoss, ref valueLoss, ref entropySum, rng);
            else
                UpdateDiscrete(network, t, alpha, config, ref policyLoss, ref valueLoss, ref entropySum);
        }

        // Single per-batch logAlpha update (batch-average gradient, not per-transition).
        // Sign: H < H_target → logAlpha increases (push toward more entropy).
        if (config.SacAutoTuneAlpha && batch.Length > 0)
        {
            var meanEntropy = entropySum / batch.Length;
            logAlpha -= config.LearningRate * (meanEntropy - targetEntropy);
            logAlpha = Math.Clamp(logAlpha, -20f, 4f);
        }

        network.SoftUpdateTargets(config.SacTau);

        var n = Math.Max(1, batch.Length);
        return new SacAsyncResult
        {
            PolicyLoss = policyLoss / n,
            ValueLoss = valueLoss / n,
            Entropy = entropySum / n,
            NewLogAlpha = logAlpha,
        };
    }

    // ── Discrete SAC update ──────────────────────────────────────────────────

    private static void UpdateDiscrete(
        SacNetwork network,
        Transition t,
        float alpha,
        RLTrainerConfig config,
        ref float policyLoss,
        ref float valueLoss,
        ref float entropySum)
    {
        // Compute target V(s') = sum_a pi(a|s') * (min_Q_target(s',a) - alpha * log_pi(a|s'))
        var (nextQ1, nextQ2) = network.GetDiscreteTargetQValues(t.NextObservation);
        var (nextProbs, nextLogProbs) = network.GetDiscreteActorProbs(t.NextObservation);

        var nextV = 0f;
        for (var a = 0; a < nextQ1.Length; a++)
        {
            var minQ = Math.Min(nextQ1[a], nextQ2[a]);
            nextV += nextProbs[a] * (minQ - alpha * nextLogProbs[a]);
        }

        // Bellman target
        var mask = t.Done ? 0f : 1f;
        var y = t.Reward + config.Gamma * mask * nextV;

        // Update Q networks
        var (q1, q2) = network.GetDiscreteQValues(t.Observation);
        valueLoss += MathF.Abs(q1[t.DiscreteAction] - y) + MathF.Abs(q2[t.DiscreteAction] - y);

        network.UpdateQ1Discrete(t.Observation, t.DiscreteAction, y);
        network.UpdateQ2Discrete(t.Observation, t.DiscreteAction, y);

        // Update actor
        network.UpdateActorDiscrete(t.Observation, alpha);

        // Compute entropy for logging; logAlpha update happens once per batch in the caller.
        var (probs, logProbs) = network.GetDiscreteActorProbs(t.Observation);
        var entropy = 0f;
        for (var a = 0; a < probs.Length; a++)
            entropy -= probs[a] * logProbs[a];

        entropySum += entropy;
        policyLoss -= entropy; // policy loss ≈ -entropy as a proxy metric
    }

    // ── Continuous SAC update ────────────────────────────────────────────────

    private static void UpdateContinuous(
        SacNetwork network,
        Transition t,
        float alpha,
        RLTrainerConfig config,
        ref float policyLoss,
        ref float valueLoss,
        ref float entropySum,
        Random rng)
    {
        // Sample next action for target
        var (nextAction, nextLogProb, _, _) = network.SampleContinuousAction(t.NextObservation, rng);
        var (nextQ1t, nextQ2t) = network.GetContinuousTargetQValues(t.NextObservation, nextAction);
        var nextMinQ = Math.Min(nextQ1t, nextQ2t);
        var nextV = nextMinQ - alpha * nextLogProb;

        var mask = t.Done ? 0f : 1f;
        var y = t.Reward + config.Gamma * mask * nextV;

        // Update Q networks
        var (q1, q2) = network.GetContinuousQValues(t.Observation, t.ContinuousActions);
        valueLoss += MathF.Abs(q1 - y) + MathF.Abs(q2 - y);

        network.UpdateQ1Continuous(t.Observation, t.ContinuousActions, y);
        network.UpdateQ2Continuous(t.Observation, t.ContinuousActions, y);

        // Sample fresh action for actor update (reparameterization)
        var (action, logProb, eps, u) = network.SampleContinuousAction(t.Observation, rng);
        network.UpdateActorContinuous(t.Observation, action, eps, u, alpha);

        // Accumulate entropy; logAlpha update happens once per batch in the caller.
        entropySum += -logProb;
        policyLoss += logProb; // policy loss proxy
    }

    // ── IDistributedTrainer ───────────────────────────────────────────────────

    public bool IsOffPolicy => true;

    public bool IsRolloutReady => _stagingTransitions.Count >= _trainerConfig.SacBatchSize;

    public byte[] ExportAndClearRollout()
    {
        var batch = new List<DistributedTransition>(_stagingTransitions.Count);
        while (_stagingTransitions.TryDequeue(out var t))
        {
            batch.Add(new DistributedTransition
            {
                Observation       = t.Observation,
                DiscreteAction    = t.DiscreteAction,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                NextObservation   = t.NextObservation,
                OldLogProbability = 0f,
                Value             = 0f,
                NextValue         = 0f,
            });
        }
        return DistributedProtocol.SerializeRollout(batch);
    }

    public void InjectRollout(byte[] data)
    {
        foreach (var t in DistributedProtocol.DeserializeRollout(data))
        {
            _buffer.Add(new Transition
            {
                Observation       = t.Observation,
                DiscreteAction    = t.DiscreteAction,
                ContinuousActions = t.ContinuousActions,
                Reward            = t.Reward,
                Done              = t.Done,
                NextObservation   = t.NextObservation,
            });
        }
    }

    public byte[] ExportWeights()
    {
        var cp = _network.SaveCheckpoint("_dist_", 0, 0, 0);
        return DistributedProtocol.SerializeWeights(cp.WeightBuffer, cp.LayerShapeBuffer);
    }

    public void ImportWeights(byte[] data)
    {
        var (weights, shapes) = DistributedProtocol.DeserializeWeights(data);
        _network.LoadCheckpoint(new RLCheckpoint { WeightBuffer = weights, LayerShapeBuffer = shapes });
    }

    // ── Private types ─────────────────────────────────────────────────────────

    private sealed class SacAsyncResult
    {
        public float PolicyLoss { get; init; }
        public float ValueLoss { get; init; }
        public float Entropy { get; init; }
        public float NewLogAlpha { get; init; }
    }
}
