using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLAgentConfig : Resource
{
    [Export] public RLAgentControlMode ControlMode { get; set; } = RLAgentControlMode.Train;
    [Export(PropertyHint.File, "*.json,*.rlmodel")] public string InferenceCheckpointPath { get; set; } = string.Empty;
    /// <summary>
    /// Agents with the same non-empty PolicyGroup share one trainer brain.
    /// Empty string means each agent gets its own brain (keyed by NodePath).
    /// </summary>
    [Export] public string PolicyGroup { get; set; } = string.Empty;
}
