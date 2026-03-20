using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
[Tool]
public partial class RLRunConfig : Resource
{
    [Export] public string RunPrefix { get; set; } = string.Empty;
    [Export] public float SimulationSpeed { get; set; } = 1.0f;
    [Export(PropertyHint.Range, "1,256,or_greater")] public int BatchSize { get; set; } = 1;
    [Export] public int ActionRepeat { get; set; } = 4;
    [Export] public int CheckpointInterval { get; set; } = 10;
    [Export] public bool ShowBatchGrid { get; set; } = false;
    /// <summary>
    /// Run PPO gradient updates on a background thread while the main thread continues
    /// collecting transitions. Eliminates PPO backprop spikes from the main-thread frame budget.
    /// Opt-in; disabled by default for predictable single-threaded behavior.
    /// </summary>
    [Export] public bool AsyncGradientUpdates { get; set; } = false;
    /// <summary>
    /// When two or more policy groups are active, run their value-estimation (Phase A) and
    /// action-sampling (Phase C) forward passes in parallel across <c>System.Threading.Tasks</c>
    /// worker threads. Phase B (episode resets, Godot API) and Phase D (ApplyDecision, Godot API)
    /// remain on the main thread. Opt-in; has no effect with a single policy group.
    /// </summary>
    [Export] public bool ParallelPolicyGroups { get; set; } = false;
}
