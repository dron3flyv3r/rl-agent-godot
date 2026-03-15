using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAcademy : Node
{
    [Export] public RLTrainerConfig? TrainerConfig { get; set; }
    [Export] public RLNetworkConfig? NetworkConfig { get; set; }
    [Export] public RLCheckpoint? Checkpoint { get; set; }

    [ExportGroup("Training")]
    [Export] public string RunPrefix { get; set; } = string.Empty;
    [Export] public int CheckpointSaveIntervalUpdates { get; set; } = 10;
    [Export] public float SimulationSpeed { get; set; } = 1.0f;

    public bool InferenceActive { get; private set; }

    private readonly Dictionary<RLAgent2D, PolicyValueNetwork> _agentInferenceNetworks = new();

    public override void _Ready()
    {
        TryInitializeInference();
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!InferenceActive)
        {
            return;
        }

        foreach (var pair in _agentInferenceNetworks)
        {
            var agent = pair.Key;
            var network = pair.Value;

            agent.TickStep();
            if (agent.ConsumeDonePending())
            {
                agent.ResetEpisode();
            }

            var observation = agent.GetObservation();
            if (observation.Length == 0)
            {
                continue;
            }

            agent.ApplyAction(network.SelectGreedyAction(observation));
        }
    }

    public IReadOnlyList<RLAgent2D> GetAgents()
    {
        var agents = new List<RLAgent2D>();
        var sceneRoot = ResolveSceneRoot();
        CollectAgents(sceneRoot, agents);
        return agents;
    }

    public IReadOnlyList<RLAgent2D> GetAgents(RLAgentControlMode controlMode)
    {
        var agents = new List<RLAgent2D>();
        foreach (var agent in GetAgents())
        {
            if (agent.ControlMode == controlMode)
            {
                agents.Add(agent);
            }
        }

        return agents;
    }

    public void ResetAllAgents()
    {
        foreach (var agent in GetAgents())
        {
            agent.ResetEpisode();
        }
    }

    private static void CollectAgents(Node node, ICollection<RLAgent2D> agents)
    {
        if (node is RLAgent2D agent)
        {
            agents.Add(agent);
        }

        foreach (var child in node.GetChildren())
        {
            if (child is Node childNode)
            {
                CollectAgents(childNode, agents);
            }
        }
    }

    private Node ResolveSceneRoot()
    {
        var current = this as Node;
        while (current.GetParent() is Node parent && parent is not TrainingBootstrap)
        {
            current = parent;
        }

        return current;
    }

    private void TryInitializeInference()
    {
        if (ResolveSceneRoot().GetParent() is TrainingBootstrap || NetworkConfig is null)
        {
            return;
        }

        var agents = GetAgents();
        if (agents.Count == 0)
        {
            return;
        }

        // Warn about any agents left in Train mode when running outside of the training bootstrap.
        foreach (var agent in agents)
        {
            if (agent.ControlMode == RLAgentControlMode.Train)
            {
                GD.PushWarning(
                    $"[RLAcademy] Agent '{agent.Name}' is set to Train mode, but the game was " +
                    "started normally. Training will not run. Use the 'Start Training' button in " +
                    "the RL Agent Plugin dock, or switch the agent to Inference mode.");
            }
        }

        string? firstLoadedCheckpointPath = null;
        foreach (var agent in agents)
        {
            if (agent.ControlMode != RLAgentControlMode.Inference)
            {
                continue;
            }

            if (!agent.SupportsOnlyDiscreteActions())
            {
                GD.PushWarning($"[RLAcademy] Agent '{agent.Name}' has continuous actions which are not supported for inference yet.");
                continue;
            }

            var checkpointPath = agent.GetInferenceCheckpointPath();

            // For .rlmodel files resolve path directly; for .json fall back to registry.
            RLCheckpoint? checkpoint;
            if (checkpointPath.EndsWith(".rlmodel", StringComparison.OrdinalIgnoreCase))
            {
                if (!FileAccess.FileExists(checkpointPath))
                {
                    GD.PushWarning($"[RLAcademy] .rlmodel not found for agent '{agent.Name}': {checkpointPath}");
                    continue;
                }

                checkpoint = RLModelLoader.LoadFromFile(checkpointPath);
            }
            else
            {
                var resolvedPath = CheckpointRegistry.ResolveCheckpointPath(checkpointPath);
                if (string.IsNullOrWhiteSpace(resolvedPath))
                {
                    continue;
                }

                checkpoint = RLCheckpoint.LoadFromFile(resolvedPath);
                firstLoadedCheckpointPath ??= resolvedPath;
            }

            if (checkpoint is null || checkpoint.LayerShapeBuffer.Length < 6)
            {
                GD.PushWarning($"[RLAcademy] Could not load checkpoint for agent '{agent.Name}'.");
                continue;
            }

            // Read obs size and action count from the checkpoint's embedded layer shapes
            // rather than calling GetObservation() at _Ready time, which may return empty
            // before OnEpisodeBegin() has had a chance to set up the scene.
            var layerCount = checkpoint.LayerShapeBuffer.Length / 3;
            var obsSize    = checkpoint.LayerShapeBuffer[0];
            var actionCount = checkpoint.LayerShapeBuffer[(layerCount - 2) * 3 + 1];

            if (obsSize <= 0 || actionCount <= 0)
            {
                GD.PushWarning($"[RLAcademy] Checkpoint for agent '{agent.Name}' has invalid dimensions (obs={obsSize}, actions={actionCount}).");
                continue;
            }

            try
            {
                var network = new PolicyValueNetwork(obsSize, actionCount, NetworkConfig);
                network.LoadCheckpoint(checkpoint);
                _agentInferenceNetworks[agent] = network;
                InferenceActive = true;
                GD.Print($"[RLAcademy] Loaded inference model for '{agent.Name}' (obs={obsSize}, actions={actionCount}).");
            }
            catch (Exception ex)
            {
                GD.PushError($"[RLAcademy] Failed to load checkpoint for agent '{agent.Name}': {ex.Message} — " +
                             "Make sure the RLNetworkConfig on the academy matches the one used during training.");
            }
        }

        if (InferenceActive && !string.IsNullOrWhiteSpace(firstLoadedCheckpointPath))
        {
            Checkpoint = GD.Load<RLCheckpoint>(firstLoadedCheckpointPath);
        }
    }
}
