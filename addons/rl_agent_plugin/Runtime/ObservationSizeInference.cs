using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

public static class ObservationSizeInference
{
    public static ObservationSizeInferenceResult Infer(Node sceneRoot, IEnumerable<RLAgent2D> agents, bool resetEpisodes = true)
    {
        var result = new ObservationSizeInferenceResult();
        var firstSizeByGroup = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var agent in agents)
        {
            var binding = RLPolicyGroupBindingResolver.Resolve(sceneRoot, agent);
            result.AgentBindings[agent] = binding;

            if (!TryInferAgentObservationSize(agent, out var observationSize, out var error, resetEpisodes))
            {
                result.Errors.Add(BuildAgentError(sceneRoot, agent, error));
                continue;
            }

            result.AgentSizes[agent] = observationSize;
            if (observationSize <= 0)
            {
                result.Errors.Add(
                    $"Group '{binding.DisplayName}': agent '{sceneRoot.GetPathTo(agent)}' did not emit a non-zero observation vector.");
                continue;
            }

            if (firstSizeByGroup.TryGetValue(binding.BindingKey, out var firstSize))
            {
                if (firstSize != observationSize)
                {
                    result.Errors.Add(
                        $"Group '{binding.DisplayName}': agent '{sceneRoot.GetPathTo(agent)}' emitted {observationSize} observations, " +
                        $"expected {firstSize}.");
                }

                continue;
            }

            firstSizeByGroup[binding.BindingKey] = observationSize;
            result.GroupSizes[binding.BindingKey] = observationSize;
        }

        return result;
    }

    public static bool TryInferAgentObservationSize(
        RLAgent2D agent,
        out int observationSize,
        out string error,
        bool resetEpisode = true)
    {
        try
        {
            if (resetEpisode)
            {
                agent.ResetEpisode();
            }

            observationSize = agent.CollectObservationArray().Length;
            error = string.Empty;
            return true;
        }
        catch (Exception exception)
        {
            observationSize = 0;
            error = exception.Message;
            return false;
        }
    }

    private static string BuildAgentError(Node sceneRoot, RLAgent2D agent, string error)
    {
        var agentPath = sceneRoot.GetPathTo(agent);
        return $"Agent '{agentPath}': observation inference failed: {error}";
    }
}
