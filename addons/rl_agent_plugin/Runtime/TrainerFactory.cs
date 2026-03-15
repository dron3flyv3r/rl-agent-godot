using System;

namespace RlAgentPlugin.Runtime;

public static class TrainerFactory
{
    public static ITrainer Create(PolicyGroupConfig config)
    {
        return config.Algorithm switch
        {
            RLAlgorithmKind.PPO => new PpoTrainer(config),
            RLAlgorithmKind.SAC => new SacTrainer(config),
            _ => throw new NotSupportedException($"Unknown algorithm: {config.Algorithm}"),
        };
    }
}
