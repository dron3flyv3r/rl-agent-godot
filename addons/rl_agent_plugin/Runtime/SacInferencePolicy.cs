using System;

namespace RlAgentPlugin.Runtime;

public sealed class SacInferencePolicy : IInferencePolicy
{
    private readonly SacNetwork _network;
    private readonly bool _isContinuous;

    public SacInferencePolicy(int observationSize, int actionDimensions, bool isContinuous, RLNetworkConfig config)
    {
        if (actionDimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(actionDimensions), "SAC inference requires at least one action dimension.");
        }

        _network = new SacNetwork(observationSize, actionDimensions, isContinuous, config, 0f);
        _isContinuous = isContinuous;
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        _network.LoadActorCheckpoint(checkpoint);
    }

    public PolicyDecision Predict(float[] observation)
    {
        return _isContinuous
            ? new PolicyDecision { ContinuousActions = _network.DeterministicContinuousAction(observation) }
            : new PolicyDecision { DiscreteAction = _network.GreedyDiscreteAction(observation) };
    }
}
