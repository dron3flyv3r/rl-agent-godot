using System;
using System.Collections.Generic;
using System.Linq;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// SAC neural networks: actor + twin Q-networks + twin target Q-networks.
/// Supports both discrete actions (Q per action) and continuous actions (Q from obs+action).
/// </summary>
internal sealed class SacNetwork
{
    private readonly int _obsSize;
    private readonly int _actionDim;
    private readonly float _learningRate;

    // Actor layers (trunk + head)
    private readonly DenseLayer[] _actorTrunk;
    private readonly DenseLayer _actorHead;

    // Twin Q-networks
    private readonly DenseLayer[] _q1Trunk;
    private readonly DenseLayer _q1Head;
    private readonly DenseLayer[] _q2Trunk;
    private readonly DenseLayer _q2Head;

    // Target Q-networks (no optimizer needed, only forward + soft-update)
    private readonly DenseLayer[] _q1TargetTrunk;
    private readonly DenseLayer _q1TargetHead;
    private readonly DenseLayer[] _q2TargetTrunk;
    private readonly DenseLayer _q2TargetHead;

    public SacNetwork(int obsSize, int actionDim, bool isContinuous, RLNetworkConfig networkConfig, float learningRate)
    {
        _obsSize = obsSize;
        _actionDim = actionDim;
        _learningRate = learningRate;
        var useAdam = networkConfig.Optimizer == RLOptimizerKind.Adam;

        var hiddenSizes = networkConfig.HiddenLayerSizes.Where(s => s > 0).ToArray();

        _actorTrunk = BuildTrunk(obsSize, hiddenSizes, useAdam, networkConfig.Activation);
        var actorTrunkOut = hiddenSizes.Length > 0 ? hiddenSizes[^1] : obsSize;
        // Discrete: logits over actions; Continuous: [mean_0..mean_D, log_std_0..log_std_D]
        var actorOutSize = isContinuous ? actionDim * 2 : actionDim;
        _actorHead = new DenseLayer(actorTrunkOut, actorOutSize, null, useAdam);

        // Q-networks
        var qInputSize = isContinuous ? obsSize + actionDim : obsSize;
        var qOutSize = isContinuous ? 1 : actionDim;

        _q1Trunk = BuildTrunk(qInputSize, hiddenSizes, useAdam, networkConfig.Activation);
        var q1TrunkOut = hiddenSizes.Length > 0 ? hiddenSizes[^1] : qInputSize;
        _q1Head = new DenseLayer(q1TrunkOut, qOutSize, null, useAdam);

        _q2Trunk = BuildTrunk(qInputSize, hiddenSizes, useAdam, networkConfig.Activation);
        _q2Head = new DenseLayer(q1TrunkOut, qOutSize, null, useAdam);

        // Target Q-networks (SGD, weights will be hard-copied from online networks)
        _q1TargetTrunk = BuildTrunk(qInputSize, hiddenSizes, false, networkConfig.Activation);
        var q1TargetTrunkOut = hiddenSizes.Length > 0 ? hiddenSizes[^1] : qInputSize;
        _q1TargetHead = new DenseLayer(q1TargetTrunkOut, qOutSize, null, false);

        _q2TargetTrunk = BuildTrunk(qInputSize, hiddenSizes, false, networkConfig.Activation);
        _q2TargetHead = new DenseLayer(q1TargetTrunkOut, qOutSize, null, false);

        // Initialize targets as hard copies of online networks
        HardCopyToTargets();
    }

    // ── Discrete action methods ──────────────────────────────────────────────

    public (int action, float logProb, float entropy) SampleDiscreteAction(float[] obs, Random rng)
    {
        var logits = ForwardActor(obs);
        var probs = Softmax(logits);
        var action = SampleFromProbs(probs, rng);
        var logProb = MathF.Log(Math.Max(probs[action], 1e-8f));
        var entropy = 0f;
        foreach (var p in probs)
        {
            if (p > 1e-8f) entropy -= p * MathF.Log(p);
        }

        return (action, logProb, entropy);
    }

    public int GreedyDiscreteAction(float[] obs)
    {
        var logits = ForwardActor(obs);
        var best = 0;
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best]) best = i;
        }

        return best;
    }

    /// <summary>
    /// Returns (probs, logProbs) for all discrete actions from the actor.
    /// </summary>
    public (float[] probs, float[] logProbs) GetDiscreteActorProbs(float[] obs)
    {
        var logits = ForwardActor(obs);
        var probs = Softmax(logits);
        var logProbs = new float[probs.Length];
        for (var i = 0; i < probs.Length; i++)
        {
            logProbs[i] = probs[i] > 1e-8f ? MathF.Log(probs[i]) : MathF.Log(1e-8f);
        }

        return (probs, logProbs);
    }

    /// <summary>
    /// Returns online Q1[a] and Q2[a] for all actions (discrete).
    /// </summary>
    public (float[] q1, float[] q2) GetDiscreteQValues(float[] obs)
    {
        return (ForwardQ(_q1Trunk, _q1Head, obs), ForwardQ(_q2Trunk, _q2Head, obs));
    }

    /// <summary>
    /// Returns target Q1[a] and Q2[a] for all actions (discrete).
    /// </summary>
    public (float[] q1Target, float[] q2Target) GetDiscreteTargetQValues(float[] obs)
    {
        return (ForwardQ(_q1TargetTrunk, _q1TargetHead, obs), ForwardQ(_q2TargetTrunk, _q2TargetHead, obs));
    }

    // ── Continuous action methods ────────────────────────────────────────────

    /// <summary>
    /// Sample continuous action via reparameterization with tanh squashing.
    /// Returns (action, logProb).
    /// </summary>
    public (float[] action, float logProb, float[] eps, float[] u) SampleContinuousAction(float[] obs, Random rng)
    {
        var actorOut = ForwardActor(obs);
        var mean = actorOut[.._actionDim];
        var logStd = new float[_actionDim];
        for (var i = 0; i < _actionDim; i++)
        {
            logStd[i] = Math.Clamp(actorOut[_actionDim + i], -20f, 2f);
        }

        var eps = new float[_actionDim];
        var u = new float[_actionDim];
        var action = new float[_actionDim];
        var logProb = 0f;
        for (var i = 0; i < _actionDim; i++)
        {
            eps[i] = SampleNormal(rng);
            var std = MathF.Exp(logStd[i]);
            u[i] = mean[i] + std * eps[i];
            action[i] = MathF.Tanh(u[i]);
            // log_pi = Normal.logpdf(eps) - log(1 - tanh^2(u) + eps_small) - log_std
            var squash = 1f - action[i] * action[i];
            logProb += -0.5f * eps[i] * eps[i] - logStd[i] - MathF.Log(squash + 1e-6f);
        }

        // Constant term: -D/2 * log(2*pi)
        logProb -= _actionDim * 0.5f * MathF.Log(2f * MathF.PI);

        return (action, logProb, eps, u);
    }

    /// <summary>
    /// Returns online Q1 and Q2 for a continuous [obs, action] input.
    /// </summary>
    public (float q1, float q2) GetContinuousQValues(float[] obs, float[] action)
    {
        var input = Concat(obs, action);
        return (ForwardQ(_q1Trunk, _q1Head, input)[0], ForwardQ(_q2Trunk, _q2Head, input)[0]);
    }

    /// <summary>
    /// Returns target Q1 and Q2 for a continuous [obs, action] input.
    /// </summary>
    public (float q1Target, float q2Target) GetContinuousTargetQValues(float[] obs, float[] action)
    {
        var input = Concat(obs, action);
        return (ForwardQ(_q1TargetTrunk, _q1TargetHead, input)[0], ForwardQ(_q2TargetTrunk, _q2TargetHead, input)[0]);
    }

    // ── Gradient updates ─────────────────────────────────────────────────────

    /// <summary>Update Q1 toward target y for a given discrete action.</summary>
    public void UpdateQ1Discrete(float[] obs, int action, float target)
    {
        var qVals = ForwardQ(_q1Trunk, _q1Head, obs);
        var grad = new float[qVals.Length]; // zero gradient for all other actions
        grad[action] = qVals[action] - target;
        BackwardQ(_q1Trunk, _q1Head, obs, grad);
    }

    /// <summary>Update Q2 toward target y for a given discrete action.</summary>
    public void UpdateQ2Discrete(float[] obs, int action, float target)
    {
        var qVals = ForwardQ(_q2Trunk, _q2Head, obs);
        var grad = new float[qVals.Length];
        grad[action] = qVals[action] - target;
        BackwardQ(_q2Trunk, _q2Head, obs, grad);
    }

    /// <summary>Update Q1 toward target y for continuous actions.</summary>
    public void UpdateQ1Continuous(float[] obs, float[] action, float target)
    {
        var input = Concat(obs, action);
        var q = ForwardQ(_q1Trunk, _q1Head, input)[0];
        BackwardQ(_q1Trunk, _q1Head, input, new[] { q - target });
    }

    /// <summary>Update Q2 toward target y for continuous actions.</summary>
    public void UpdateQ2Continuous(float[] obs, float[] action, float target)
    {
        var input = Concat(obs, action);
        var q = ForwardQ(_q2Trunk, _q2Head, input)[0];
        BackwardQ(_q2Trunk, _q2Head, input, new[] { q - target });
    }

    /// <summary>
    /// Update actor for discrete SAC: maximize E_pi[min_Q(s,a) - alpha * log_pi(a|s)].
    /// Gradient of logit j: -pi(j) * (f(j) - E_pi[f]) where f(a) = min_Q(s,a) - alpha * log_pi(a).
    /// </summary>
    public void UpdateActorDiscrete(float[] obs, float alpha)
    {
        var logits = ForwardActorFull(obs, out var trunkCaches);
        var probs = Softmax(logits);

        var (q1, q2) = GetDiscreteQValues(obs);
        var logProbClipped = new float[probs.Length];
        for (var i = 0; i < probs.Length; i++)
        {
            logProbClipped[i] = probs[i] > 1e-8f ? MathF.Log(probs[i]) : MathF.Log(1e-8f);
        }

        // f(a) = min_Q(s,a) - alpha * log_pi(a)
        var f = new float[probs.Length];
        var eFpi = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            f[i] = Math.Min(q1[i], q2[i]) - alpha * logProbClipped[i];
            eFpi += probs[i] * f[i];
        }

        // Actor gradient of loss = -V: grad_j = -pi(j) * (f(j) - E_pi[f])
        var grad = new float[probs.Length];
        for (var j = 0; j < probs.Length; j++)
        {
            grad[j] = -probs[j] * (f[j] - eFpi);
        }

        BackwardActorFromLogits(trunkCaches, logits, grad);
    }

    /// <summary>
    /// Update actor for continuous SAC via reparameterization gradient.
    /// </summary>
    public void UpdateActorContinuous(float[] obs, float[] action, float[] eps, float[] u, float alpha)
    {
        // Compute dQ/d(action) without updating Q network weights
        var input = Concat(obs, action);
        var q1 = ForwardQ(_q1Trunk, _q1Head, input)[0];
        var q2 = ForwardQ(_q2Trunk, _q2Head, input)[0];

        float[] qInputGrad;
        if (q1 <= q2)
        {
            ForwardQFull(_q1Trunk, _q1Head, input, out var q1Caches);
            qInputGrad = ComputeQInputGradOnly(_q1Trunk, _q1Head, input, q1Caches, new[] { 1f });
        }
        else
        {
            ForwardQFull(_q2Trunk, _q2Head, input, out var q2Caches);
            qInputGrad = ComputeQInputGradOnly(_q2Trunk, _q2Head, input, q2Caches, new[] { 1f });
        }

        // Action gradient from Q (slice off the obs part)
        var dQdAction = qInputGrad[_obsSize..];

        // Actor outputs [mean_0..mean_{D-1}, log_std_0..log_std_{D-1}]
        var actorOut = ForwardActorFull(obs, out var actorCaches);
        var actorGrad = new float[_actionDim * 2];

        for (var d = 0; d < _actionDim; d++)
        {
            var a = action[d];
            var squash = 1f - a * a; // sech^2(u) = 1 - tanh^2(u)
            var logStd = Math.Clamp(actorOut[_actionDim + d], -20f, 2f);
            var std = MathF.Exp(logStd);

            // Gradient of loss = -V = -(min_Q - alpha * log_pi) w.r.t. actor outputs.
            //
            // d(-V)/d(mean_d) = -dQ/da_d * squash + alpha * d(-log_pi)/d(mean_d)
            //   where d(-log_pi)/d(mean_d) = 2*a*squash/(squash+eps) via chain rule through tanh
            //   but NOTE: log_pi = -log(sech^2) - log_std - ... so d(log_pi)/d(mean) = 2*a*squash/(squash+eps)
            //   and d(-log_pi)/d(mean) = -2*a*squash/(squash+eps) [increases entropy → decreases -log_pi]
            //   SAC maximizes V = Q + alpha*H, so grad of -V w.r.t. mean:
            //   = -dQ/da * squash - alpha * d(log_pi)/d(mean)
            //   = -dQdAction[d]*squash - alpha * 2*a*squash/(squash+eps)
            //   Wait, H = -E[log_pi], so dV/d(mean) = dQ/da*squash - alpha*d(log_pi)/d(mean)
            //   d(log_pi)/d(mean) = 2*a*squash/(squash+eps) (via chain rule, log(1-a^2) term)
            //   d(-V)/d(mean) = -dQ/da*squash + alpha*d(log_pi)/d(mean)
            //                 = squash*(-dQdAction[d] + 2*alpha*a/(squash+eps))
            actorGrad[d] = squash * (-dQdAction[d] + 2f * alpha * a / (squash + 1e-6f));

            // d(-V)/d(log_std_d) = -dQ/da_d * squash * std * eps_d + alpha * d(log_pi)/d(log_std_d)
            //   d(log_pi)/d(log_std_d) = -1 + 2*a*squash*std*eps_d/(squash+eps)
            //   (the -1 comes from the -log_std term in log_pi)
            //   d(-V)/d(log_std_d) = -dQdAction[d]*squash*std*eps_d - alpha*(1 - 2*a*squash*std*eps_d/(squash+eps))
            actorGrad[_actionDim + d] = -dQdAction[d] * squash * std * eps[d]
                - alpha * (1f - 2f * a * squash * std * eps[d] / (squash + 1e-6f));
        }

        BackwardActorFromLogits(actorCaches, actorOut, actorGrad);
    }

    /// <summary>Soft-updates target networks: θ_target = τ*θ + (1-τ)*θ_target.</summary>
    public void SoftUpdateTargets(float tau)
    {
        for (var i = 0; i < _q1Trunk.Length; i++)
        {
            _q1TargetTrunk[i].SoftUpdateFrom(_q1Trunk[i], tau);
            _q2TargetTrunk[i].SoftUpdateFrom(_q2Trunk[i], tau);
        }

        _q1TargetHead.SoftUpdateFrom(_q1Head, tau);
        _q2TargetHead.SoftUpdateFrom(_q2Head, tau);
    }

    // ── Checkpoint ───────────────────────────────────────────────────────────

    public RLCheckpoint SaveCheckpoint(string groupId, long totalSteps, long episodeCount, long updateCount)
    {
        var weights = new List<float>();
        var shapes = new List<int>();

        foreach (var layer in _actorTrunk) layer.AppendSerialized(weights, shapes);
        _actorHead.AppendSerialized(weights, shapes);
        foreach (var layer in _q1Trunk) layer.AppendSerialized(weights, shapes);
        _q1Head.AppendSerialized(weights, shapes);
        foreach (var layer in _q2Trunk) layer.AppendSerialized(weights, shapes);
        _q2Head.AppendSerialized(weights, shapes);

        return new RLCheckpoint
        {
            RunId = groupId,
            TotalSteps = totalSteps,
            EpisodeCount = episodeCount,
            UpdateCount = updateCount,
            WeightBuffer = weights.ToArray(),
            LayerShapeBuffer = shapes.ToArray(),
        };
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        var wi = 0;
        var si = 0;
        foreach (var layer in _actorTrunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        _actorHead.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        foreach (var layer in _q1Trunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        _q1Head.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        foreach (var layer in _q2Trunk) layer.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);
        _q2Head.LoadSerialized(checkpoint.WeightBuffer, ref wi, checkpoint.LayerShapeBuffer, ref si);

        // Reconstruct targets from online networks
        HardCopyToTargets();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private float[] ForwardActor(float[] obs)
    {
        var x = obs;
        foreach (var layer in _actorTrunk)
        {
            x = layer.Forward(x).Activated;
        }

        return _actorHead.Forward(x).Activated;
    }

    private float[] ForwardActorFull(float[] obs, out (LayerCache[] trunk, LayerCache head, float[] trunkOut) caches)
    {
        var trunkCaches = new LayerCache[_actorTrunk.Length];
        var x = obs;
        for (var i = 0; i < _actorTrunk.Length; i++)
        {
            trunkCaches[i] = _actorTrunk[i].Forward(x);
            x = trunkCaches[i].Activated;
        }

        var headCache = _actorHead.Forward(x);
        caches = (trunkCaches, headCache, x);
        return headCache.Activated;
    }

    private void BackwardActorFromLogits(
        (LayerCache[] trunk, LayerCache head, float[] trunkOut) caches,
        float[] headActivated, float[] outputGrad)
    {
        var grad = _actorHead.Backward(caches.trunkOut, outputGrad, _learningRate);
        for (var i = _actorTrunk.Length - 1; i >= 0; i--)
        {
            // LayerCache.Input stores the input that was passed to Forward() for this layer
            grad = _actorTrunk[i].Backward(caches.trunk[i].Input, grad, _learningRate, caches.trunk[i].PreActivation);
        }
    }

    private static float[] ForwardQ(DenseLayer[] trunk, DenseLayer head, float[] input)
    {
        var x = input;
        foreach (var layer in trunk)
        {
            x = layer.Forward(x).Activated;
        }

        return head.Forward(x).Activated;
    }

    private float[] ForwardQFull(DenseLayer[] trunk, DenseLayer head, float[] input,
        out (LayerCache[] trunk, float[] trunkOut) caches)
    {
        var trunkCaches = new LayerCache[trunk.Length];
        var x = input;
        for (var i = 0; i < trunk.Length; i++)
        {
            trunkCaches[i] = trunk[i].Forward(x);
            x = trunkCaches[i].Activated;
        }

        var headOut = head.Forward(x).Activated;
        caches = (trunkCaches, x);
        return headOut;
    }

    private void BackwardQ(DenseLayer[] trunk, DenseLayer head, float[] input, float[] outputGrad)
    {
        // Re-forward to get caches
        ForwardQFull(trunk, head, input, out var caches);
        var grad = head.Backward(caches.trunkOut, outputGrad, _learningRate);
        for (var i = trunk.Length - 1; i >= 0; i--)
        {
            grad = trunk[i].Backward(caches.trunk[i].Input, grad, _learningRate, caches.trunk[i].PreActivation);
        }
    }

    /// <summary>
    /// Computes input gradient through a Q network WITHOUT updating any weights.
    /// Used for actor update via reparameterization (we need dQ/da but must not touch Q).
    /// </summary>
    private static float[] ComputeQInputGradOnly(DenseLayer[] trunk, DenseLayer head, float[] input,
        (LayerCache[] trunk, float[] trunkOut) caches, float[] outputGrad)
    {
        var grad = head.ComputeInputGrad(caches.trunkOut, outputGrad);
        for (var i = trunk.Length - 1; i >= 0; i--)
        {
            grad = trunk[i].ComputeInputGrad(caches.trunk[i].Input, grad, caches.trunk[i].PreActivation);
        }

        return grad;
    }

    private void HardCopyToTargets()
    {
        for (var i = 0; i < _q1Trunk.Length; i++)
        {
            _q1TargetTrunk[i].CopyFrom(_q1Trunk[i]);
            _q2TargetTrunk[i].CopyFrom(_q2Trunk[i]);
        }

        _q1TargetHead.CopyFrom(_q1Head);
        _q2TargetHead.CopyFrom(_q2Head);
    }

    private static DenseLayer[] BuildTrunk(int inputSize, int[] hiddenSizes, bool useAdam, RLActivationKind activation)
    {
        if (hiddenSizes.Length == 0) return Array.Empty<DenseLayer>();

        var layers = new DenseLayer[hiddenSizes.Length];
        var prev = inputSize;
        for (var i = 0; i < hiddenSizes.Length; i++)
        {
            layers[i] = new DenseLayer(prev, hiddenSizes[i], activation, useAdam);
            prev = hiddenSizes[i];
        }

        return layers;
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var probs = new float[logits.Length];
        var total = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - max);
            total += probs[i];
        }

        for (var i = 0; i < probs.Length; i++) probs[i] /= total;
        return probs;
    }

    private static int SampleFromProbs(float[] probs, Random rng)
    {
        var roll = rng.NextSingle();
        var cumulative = 0f;
        for (var i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (roll <= cumulative) return i;
        }

        return probs.Length - 1;
    }

    private static float SampleNormal(Random rng)
    {
        // Box-Muller transform
        var u1 = Math.Max(rng.NextSingle(), 1e-10f);
        var u2 = rng.NextSingle();
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }

    private static float[] Concat(float[] a, float[] b)
    {
        var result = new float[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }
}
