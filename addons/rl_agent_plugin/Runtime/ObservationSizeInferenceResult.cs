using System.Collections.Generic;

namespace RlAgentPlugin.Runtime;

public sealed class ObservationSizeInferenceResult
{
    public Dictionary<RLAgent2D, int> AgentSizes { get; } = new();
    public Dictionary<RLAgent2D, ResolvedPolicyGroupBinding> AgentBindings { get; } = new();
    public Dictionary<string, int> GroupSizes { get; } = new(System.StringComparer.Ordinal);
    public List<string> Errors { get; } = new();

    public bool IsValid => Errors.Count == 0;
}
