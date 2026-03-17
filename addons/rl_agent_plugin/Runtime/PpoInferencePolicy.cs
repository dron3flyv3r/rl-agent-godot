using System;

namespace RlAgentPlugin.Runtime;

public sealed class PpoInferencePolicy : IInferencePolicy
{
    private readonly PolicyValueNetwork _network;

    public PpoInferencePolicy(int observationSize, int actionCount, RLNetworkGraph graph)
    {
        if (actionCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(actionCount), "PPO inference requires at least one discrete action.");
        }

        _network = new PolicyValueNetwork(observationSize, actionCount, graph);
    }

    public void LoadCheckpoint(RLCheckpoint checkpoint)
    {
        _network.LoadCheckpoint(checkpoint);
    }

    public PolicyDecision Predict(float[] observation)
    {
        return new PolicyDecision
        {
            DiscreteAction = _network.SelectGreedyAction(observation),
        };
    }
}
