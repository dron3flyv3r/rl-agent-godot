using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLAgentConfig : Resource
{
    [Export] public RLAgentControlMode ControlMode { get; set; } = RLAgentControlMode.Train;
    [Export(PropertyHint.File, "*.json,*.rlmodel")] public string InferenceCheckpointPath { get; set; } = string.Empty;

    /// <summary>
    /// Explicit shared-policy resource. When assigned, runtime/editor grouping should prefer
    /// this resource over the legacy PolicyGroup string.
    /// </summary>
    [Export] public RLPolicyGroupConfig? PolicyGroupConfig { get; set; }

    /// <summary>
    /// Legacy fallback grouping key. Prefer PolicyGroupConfig for new scenes.
    /// Agents with the same non-empty PolicyGroup share one trainer brain.
    /// Empty string means each agent gets its own brain (keyed by NodePath).
    /// </summary>
    [Export] public string PolicyGroup { get; set; } = string.Empty;
}
