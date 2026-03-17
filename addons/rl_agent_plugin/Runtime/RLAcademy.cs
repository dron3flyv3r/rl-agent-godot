using System;
using System.Collections.Generic;
using Godot;

namespace RlAgentPlugin.Runtime;

[GlobalClass]
public partial class RLAcademy : Node
{
    private RLTrainingConfig? _trainingConfig;

    [ExportGroup("Configuration")]
    [Export]
    public RLTrainingConfig? TrainingConfig
    {
        get => _trainingConfig;
        set { _trainingConfig = value; UpdateConfigurationWarnings(); }
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
        // if (TrainingConfig is not null && (TrainerConfig is not null || NetworkConfig is not null))
        //     warnings.Add("TrainingConfig is assigned. Legacy TrainerConfig and NetworkConfig are ignored until they are migrated.");
        if (TrainingConfig is null && ResolveTrainerConfig() is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource or a legacy RLTrainerConfig resource.");
        if (TrainingConfig is null && ResolveNetworkConfig() is null)
            warnings.Add("TrainingConfig is not assigned. Assign an RLTrainingConfig resource or a legacy RLNetworkConfig resource.");
        return warnings.ToArray();
    }

    private readonly Dictionary<RLAgent2D, IInferencePolicy> _agentInferencePolicies = new();
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

        foreach (var pair in _agentInferencePolicies)
        {
            var agent = pair.Key;
            var policy = pair.Value;

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

                var decision = policy.Predict(observation);
                if (decision.DiscreteAction >= 0)
                {
                    agent.ApplyAction(decision.DiscreteAction);
                }
                else if (decision.ContinuousActions.Length > 0)
                {
                    agent.ApplyAction(decision.ContinuousActions);
                }
            }
            else
            {
                // Re-apply current action so physics-driven movement continues each tick.
                if (agent.CurrentActionIndex >= 0)
                {
                    agent.ApplyAction(agent.CurrentActionIndex);
                }
                else if (agent.CurrentContinuousActions.Length > 0)
                {
                    agent.ApplyAction(agent.CurrentContinuousActions);
                }
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
        if (ResolveSceneRoot().GetParent() is TrainingBootstrap)
        {
            return;
        }

        var fallbackNetworkConfig = ResolveNetworkConfig();
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

        RLCheckpoint? firstLoadedCheckpoint = null;
        foreach (var agent in agents)
        {
            if (agent.ControlMode != RLAgentControlMode.Inference)
            {
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
            }

            if (checkpoint is null)
            {
                GD.PushWarning($"[RLAcademy] Could not load checkpoint for agent '{agent.Name}'.");
                continue;
            }

            var obsSize = checkpoint.ObservationSize;
            var actionCount = checkpoint.DiscreteActionCount > 0
                ? checkpoint.DiscreteActionCount
                : checkpoint.ContinuousActionDimensions;
            var agentObsSize = agent.GetExpectedObservationSize();
            var agentDiscreteCount = agent.GetDiscreteActionCount();
            var agentContinuousDims = agent.GetContinuousActionDimensions();

            if (obsSize != agentObsSize)
            {
                GD.PushError(
                    $"[RLAcademy] Checkpoint observation size {obsSize} does not match agent '{agent.Name}' " +
                    $"observation size {agentObsSize}.");
                continue;
            }

            if (string.Equals(checkpoint.Algorithm, RLCheckpoint.PpoAlgorithm, StringComparison.OrdinalIgnoreCase)
                && agentContinuousDims > 0)
            {
                GD.PushError($"[RLAcademy] Checkpoint for '{agent.Name}' is PPO, but the agent expects continuous actions.");
                continue;
            }

            if (checkpoint.ContinuousActionDimensions > 0 && checkpoint.ContinuousActionDimensions != agentContinuousDims)
            {
                GD.PushError(
                    $"[RLAcademy] Continuous action mismatch for '{agent.Name}': " +
                    $"checkpoint={checkpoint.ContinuousActionDimensions}, agent={agentContinuousDims}.");
                continue;
            }

            if (checkpoint.DiscreteActionCount > 0 && checkpoint.DiscreteActionCount != agentDiscreteCount)
            {
                GD.PushError(
                    $"[RLAcademy] Discrete action mismatch for '{agent.Name}': " +
                    $"checkpoint={checkpoint.DiscreteActionCount}, agent={agentDiscreteCount}.");
                continue;
            }

            if (obsSize <= 0 || actionCount <= 0)
            {
                GD.PushWarning($"[RLAcademy] Checkpoint for agent '{agent.Name}' has invalid dimensions (obs={obsSize}, actions={actionCount}).");
                continue;
            }

            try
            {
                var policy = InferencePolicyFactory.Create(checkpoint, fallbackNetworkConfig);
                policy.LoadCheckpoint(checkpoint);
                _agentInferencePolicies[agent] = policy;
                _inferenceStepCounters[agent] = 0;
                firstLoadedCheckpoint ??= checkpoint;
                GD.Print(
                    $"[RLAcademy] Loaded {checkpoint.Algorithm} inference model for '{agent.Name}' " +
                    $"(obs={obsSize}, actions={actionCount}).");
            }
            catch (Exception ex)
            {
                GD.PushError($"[RLAcademy] Failed to load checkpoint for agent '{agent.Name}': {ex.Message} — " +
                             "Verify that the checkpoint metadata matches the active agent.");
            }
        }

        InferenceActive = _agentInferencePolicies.Count > 0;
        if (InferenceActive && firstLoadedCheckpoint is not null)
        {
            Checkpoint = firstLoadedCheckpoint;
        }
    }

    public RLTrainerConfig? ResolveTrainerConfig()
    {
        return TrainingConfig?.ToTrainerConfig();
    }

    public RLNetworkConfig? ResolveNetworkConfig()
    {
        return TrainingConfig?.ToNetworkConfig();
    }
}
