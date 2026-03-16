using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAcademy : Node
{
    private RLTrainingConfig? _trainingConfig;
    private RLTrainerConfig? _trainerConfig;
    private RLNetworkConfig? _networkConfig;

    [ExportGroup("Configuration")]
    [Export]
    public RLTrainingConfig? TrainingConfig
    {
        get => _trainingConfig;
        set { _trainingConfig = value; UpdateConfigurationWarnings(); }
    }

    [Export(PropertyHint.ResourceType, "RLTrainerConfig")]
    public RLTrainerConfig? TrainerConfig
    {
        get => _trainerConfig;
        set { _trainerConfig = value; UpdateConfigurationWarnings(); }
    }

    [Export(PropertyHint.ResourceType, "RLNetworkConfig")]
    public RLNetworkConfig? NetworkConfig
    {
        get => _networkConfig;
        set { _networkConfig = value; UpdateConfigurationWarnings(); }
    }

    [ExportGroup("Training Run")]
    [Export] public string RunPrefix { get; set; } = string.Empty;
    [Export] public float SimulationSpeed { get; set; } = 1.0f;
    // Number of gradient update steps between checkpoint saves
    [Export] public int CheckpointInterval { get; set; } = 10;
    // How many physics ticks to repeat each action before sampling the next one (1 = every tick)
    [Export] public int ActionRepeat { get; set; } = 1;
    // Number of parallel scene instances to run during training (vectorised environments)
    [Export(PropertyHint.Range, "1, 256, 8, or_greater")] public int BatchSize { get; set; } = 1;

    [ExportGroup("Inference")]
    [Export] public RLCheckpoint? Checkpoint { get; set; }

    public bool InferenceActive { get; private set; }

    public override string[] _GetConfigurationWarnings()
    {
        var warnings = new List<string>();
        if (TrainingConfig is not null && (TrainerConfig is not null || NetworkConfig is not null))
            warnings.Add("TrainingConfig is assigned. Legacy TrainerConfig and NetworkConfig are ignored until they are migrated.");
        if (TrainingConfig is null && ResolveTrainerConfig() is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource or a legacy RLTrainerConfig resource.");
        if (TrainingConfig is null && ResolveNetworkConfig() is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource or a legacy RLNetworkConfig resource.");
        return warnings.ToArray();
    }

    private readonly Dictionary<RLAgent2D, PolicyValueNetwork> _agentInferenceNetworks = new();
    private readonly Dictionary<RLAgent2D, int> _inferenceStepCounters = new();
    private double _previousTimeScale = 1.0;
    private int _previousMaxPhysicsStepsPerFrame = 8;

    public override void _Ready()
    {
        TryInitializeInference();

        // Apply simulation speed when running outside of TrainingBootstrap (which manages it on its own).
        if (ResolveSceneRoot().GetParent() is not TrainingBootstrap)
        {
            _previousTimeScale = Engine.TimeScale;
            _previousMaxPhysicsStepsPerFrame = Engine.MaxPhysicsStepsPerFrame;
            Engine.TimeScale = Math.Max(0.1f, SimulationSpeed);
            Engine.MaxPhysicsStepsPerFrame = Math.Max(8, (int)Math.Ceiling(SimulationSpeed) + 1);
        }
    }

    public override void _ExitTree()
    {
        if (ResolveSceneRoot().GetParent() is not TrainingBootstrap)
        {
            Engine.TimeScale = _previousTimeScale;
            Engine.MaxPhysicsStepsPerFrame = _previousMaxPhysicsStepsPerFrame;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (!InferenceActive)
        {
            return;
        }

        var repeat = Math.Max(1, ActionRepeat);

        foreach (var pair in _agentInferenceNetworks)
        {
            var agent = pair.Key;
            var network = pair.Value;

            agent.TickStep();
            if (agent.ConsumeDonePending())
            {
                agent.ResetEpisode();
                _inferenceStepCounters[agent] = 0;
            }

            _inferenceStepCounters.TryGetValue(agent, out var stepCount);
            stepCount++;
            _inferenceStepCounters[agent] = stepCount;

            if (stepCount >= repeat)
            {
                _inferenceStepCounters[agent] = 0;
                var observation = agent.CollectObservationArray();
                if (observation.Length == 0)
                {
                    continue;
                }

                agent.ApplyAction(network.SelectGreedyAction(observation));
            }
            else
            {
                // Re-apply current action so physics-driven movement continues each tick.
                agent.ApplyAction(agent.CurrentActionIndex);
            }
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
        var networkConfig = ResolveNetworkConfig();
        if (ResolveSceneRoot().GetParent() is TrainingBootstrap || networkConfig is null)
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
            // rather than calling CollectObservationArray() at _Ready time, which may return empty
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
                var network = new PolicyValueNetwork(obsSize, actionCount, networkConfig);
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

    public RLTrainerConfig? ResolveTrainerConfig()
    {
        return TrainingConfig?.ToTrainerConfig() ?? TrainerConfig;
    }

    public RLNetworkConfig? ResolveNetworkConfig()
    {
        return TrainingConfig?.ToNetworkConfig() ?? NetworkConfig;
    }
}
